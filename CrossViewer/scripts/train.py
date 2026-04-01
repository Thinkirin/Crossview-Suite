"""
Training script for CrossViewer
Supports multi-GPU training with DDP
"""
import argparse
import os
import sys
import json
import random
import time
import numpy as np
import itertools
import threading
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from crossviewer.config_utils import load_config, validate_required_paths
from crossviewer.model import CrossViewerModel
from data.jsonl_dataset import CrossViewerJSONLDataset, collate_fn

_MODEL_REGISTRY = {
    'CrossViewerModel': CrossViewerModel,
}

def _get_model_class(config):
    name = config.get('model', {}).get('model_class', 'CrossViewerModel')
    if name not in _MODEL_REGISTRY:
        if name == 'CrossViewerModelNoART':
            from crossviewer.model_ablation_no_art import CrossViewerModelNoART
            _MODEL_REGISTRY[name] = CrossViewerModelNoART
        elif name == 'CrossViewerModelNoCrossAttn':
            from crossviewer.model_ablation_no_crossattn import CrossViewerModelNoCrossAttn
            _MODEL_REGISTRY[name] = CrossViewerModelNoCrossAttn
        else:
            raise ValueError(f"Unknown model_class: {name}")
    return _MODEL_REGISTRY[name]


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


class Trainer:
    """Main trainer class"""

    def __init__(self, config, rank=0, world_size=1, use_deepspeed=False, deepspeed_config=None):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config = deepspeed_config
        self.is_main_process = (rank == 0)
        self.grad_accum_steps = int(self.config['training'].get('gradient_accumulation_steps', 1))
        self.disable_infonce = float(self.config['loss'].get('info_nce_weight', 1.0)) <= 0.0

        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])

        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_dataloaders()

        if self.use_deepspeed:
            self._init_deepspeed()
        else:
            if self.world_size > 1:
                self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()

        if self.is_main_process:
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            model_ref.count_parameters()
            total_steps = len(self.train_loader) * self.config['training']['num_epochs']
            warmup_steps = self.config['training'].get('warmup_steps')
            warmup_ratio = self.config['training'].get('warmup_ratio')
            if warmup_steps is None and warmup_ratio is not None:
                warmup_steps = int(total_steps * float(warmup_ratio))
            if warmup_steps is None:
                warmup_steps = 0
            print(f"✓ LR schedule: type={self.config['training'].get('lr_scheduler_type','cosine')} "
                  f"base_lr={self.config['training']['learning_rate']} "
                  f"warmup_steps={warmup_steps} total_steps={total_steps}")
            if self.disable_infonce:
                print("✓ InfoNCE disabled by config (loss.info_nce_weight <= 0).", flush=True)

        if self.is_main_process:
            self.writer = None
        self.report_to = config['training'].get('report_to', 'tensorboard')
        if isinstance(self.report_to, str):
            self.report_to = [r.strip() for r in self.report_to.split(',') if r.strip()]
        if 'none' in self.report_to:
            self.report_to = []

        if self.is_main_process and 'tensorboard' in self.report_to:
            log_dir = config['training']['log_dir']
            run_name = config['training'].get('run_name')
            if run_name:
                log_dir = os.path.join(log_dir, run_name)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"✓ TensorBoard logging to {log_dir}")
        self._init_reporters()
        self.loss_history = []

        self.global_step = 0
        self.epoch = 0
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.resume_step_in_epoch = 0

        self._maybe_resume()

    def _build_model(self):
        """Build and wrap model"""
        model_cls = _get_model_class(self.config)
        model = model_cls(
            vision_encoder_path=self.config['model']['vision_encoder_path'],
            freeze_vision_encoder=self.config['model']['freeze_vision_encoder'],
            num_object_tokens=self.config['model']['num_object_tokens'],
            num_cross_attn_heads=self.config['model']['num_cross_attn_heads'],
            contrast_dim=self.config['model']['contrast_dim'],
            temperature=self.config['loss']['temperature'],
            infonce_weight=self.config['loss'].get('info_nce_weight', 1.0),
            vqa_weight=self.config['loss'].get('vqa_weight', 0.5),
            triplet_weight=self.config['loss'].get('triplet_weight', 0.0),
            attn_implementation=self.config['model'].get('attn_implementation'),
            load_device=self.rank if self.config['model'].get('load_directly_to_gpu', False) else None,
            low_cpu_mem_usage=self.config['model'].get('low_cpu_mem_usage', True),
            match_mode=self.config['model'].get('match_mode', 'gt'),
            region_source=self.config['model'].get('region_source', 'fused'),
            pixelrefer_mode=self.config['model'].get('pixelrefer_mode', 'full'),
            preprocess_on_gpu=self.config['model'].get('preprocess_on_gpu', False),
            region_placeholder=self.config['data'].get('region_placeholder', "<region>"),
            mask_num=self.config['data'].get('mask_num', 32),
            debug_pixelrefer=self.config['model'].get('debug_pixelrefer', False),
            debug_pixelrefer_max_steps=self.config['model'].get('debug_pixelrefer_max_steps', 1),
            debug_pixelrefer_max_masks=self.config['model'].get('debug_pixelrefer_max_masks', 50),
            debug_pixelrefer_every=self.config['model'].get('debug_pixelrefer_every', 1),
            debug_pixelrefer_per_mask_preprocess=self.config['model'].get('debug_pixelrefer_per_mask_preprocess', False),
            debug_match=self.config['model'].get('debug_match', False),
            debug_max_steps=self.config['model'].get('debug_max_steps', 1),
            debug_nan=self.config['model'].get('debug_nan', False),
            freeze_pos_encoder=self.config['model'].get('freeze_pos_encoder', False),
            unfreeze_lm_head=self.config['model'].get('unfreeze_lm_head', False),
        )

        if not getattr(model, "_loaded_to_device", False):
            model = model.to(self.rank)

        return model

    def _resolve_compute_infonce(self, batch):
        """Resolve per-sample InfoNCE flags with config overrides."""
        questions = batch.get('questions') or []
        n = len(questions)
        if self.disable_infonce:
            return [False] * n

        raw_flags = batch.get('compute_infonce')
        if raw_flags is None:
            flags = [False] * n
        else:
            flags = [bool(v) for v in raw_flags]
            if len(flags) < n:
                flags = flags + [False] * (n - len(flags))
            elif len(flags) > n:
                flags = flags[:n]

        if self.config['model'].get('force_compute_infonce', False):
            flags = [True] * n
            metas = batch.get("metadata") or []
            for i, meta in enumerate(metas[:n]):
                qtype = str((meta or {}).get("question_type", "")).upper()
                if qtype in ("Q2", "Q9"):
                    flags[i] = False
        return flags

    def _init_deepspeed(self):
        """Initialize DeepSpeed engine and optimizer/scheduler"""
        try:
            import deepspeed
        except ImportError:
            raise RuntimeError("DeepSpeed is enabled in config but not installed.")

        ds_config = self.deepspeed_config
        if isinstance(ds_config, str):
            ds_path = Path(ds_config)
            if not ds_path.is_absolute():
                ds_path = Path(__file__).parent.parent / ds_path
            with open(ds_path, "r") as f:
                ds_config = json.load(f)
        if isinstance(ds_config, dict):
            ds_config.setdefault("train_micro_batch_size_per_gpu", self.config['training']['batch_size'])
            ds_config.setdefault("gradient_accumulation_steps", self.grad_accum_steps)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(params_override=params)
        scheduler = self._build_scheduler(optimizer_override=optimizer)

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=params,
            config=ds_config,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        self.model = model_engine
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _build_dataloaders(self):
        """Build train and val dataloaders"""
        cfg = self.config['data']

        train_dataset = CrossViewerJSONLDataset(
            jsonl_path=cfg['jsonl_train'],
            data_root=cfg['data_root'],
            processor_path=self.config['model']['vision_encoder_path'],
            split='train',
            max_samples=cfg.get('max_samples_train'),
            lazy_load=cfg.get('lazy_load', True),
            index_cache=cfg.get('index_cache', True),
            include_options_in_question=cfg.get('include_options_in_question', True),
            resample_strategy=cfg.get('resample_strategy', 'random'),
            image_only=cfg.get('image_only', False),
            use_additional_inputs=cfg.get('use_additional_inputs', True),
            use_processor=cfg.get('use_processor', True),
            return_raw_images=cfg.get('return_raw_images', False),
            defer_region_token_expansion=cfg.get('defer_region_token_expansion', False),
            debug_timing=cfg.get('debug_timing', False),
            timing_every=cfg.get('timing_every', 200),
            timing_max_logs=cfg.get('timing_max_logs', 50),
            debug_trace=cfg.get('debug_trace', False),
            debug_trace_max=cfg.get('debug_trace_max', 20),
            debug_trace_every=cfg.get('debug_trace_every', 1),
            max_objects_per_sample=cfg.get('max_objects_per_sample', -1),
            load_masks=True,
            prompt_template=cfg.get('prompt_template'),
            max_retries=cfg.get('max_retries', 20),
            load_timeout_sec=cfg.get('load_timeout_sec', 30),
            video_backend=cfg.get('video_backend', 'decord'),
            decord_num_threads=cfg.get('decord_num_threads', 1),
            mask_jitter=cfg.get('mask_jitter', False),
            mask_jitter_prob=cfg.get('mask_jitter_prob', 0.3),
            mask_jitter_max_shift=cfg.get('mask_jitter_max_shift', 10),
            mask_jitter_max_kernel=cfg.get('mask_jitter_max_kernel', 5),
            mask_num=cfg.get('mask_num', 32),
            region_token_num=cfg.get('region_token_num'),
            region_placeholder=cfg.get('region_placeholder', "<region>")
        )

        val_dataset = CrossViewerJSONLDataset(
            jsonl_path=cfg['jsonl_val'],
            data_root=cfg['data_root'],
            processor_path=self.config['model']['vision_encoder_path'],
            split='val',
            max_samples=cfg.get('max_samples_val'),
            lazy_load=cfg.get('lazy_load', True),
            index_cache=cfg.get('index_cache', True),
            include_options_in_question=cfg.get('include_options_in_question', True),
            load_masks=True,
            prompt_template=cfg.get('prompt_template'),
            max_retries=cfg.get('max_retries', 20),
            load_timeout_sec=cfg.get('load_timeout_sec', 30),
            video_backend=cfg.get('video_backend', 'decord'),
            decord_num_threads=cfg.get('decord_num_threads', 1),
            mask_jitter=False,
            mask_num=cfg.get('mask_num', 32),
            region_token_num=cfg.get('region_token_num'),
            region_placeholder=cfg.get('region_placeholder', "<region>"),
            use_additional_inputs=cfg.get('use_additional_inputs', True),
            use_processor=cfg.get('use_processor', True),
            return_raw_images=cfg.get('return_raw_images', False),
            defer_region_token_expansion=cfg.get('defer_region_token_expansion', False),
            max_objects_per_sample=cfg.get('max_objects_per_sample', -1),
        )

        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        num_workers = int(cfg.get('num_workers', 0))
        prefetch_factor = cfg.get('prefetch_factor', 2)
        persistent_workers = bool(cfg.get('persistent_workers', False))
        pin_memory = bool(cfg.get('pin_memory', True))

        def _worker_init_fn(_):
            try:
                import cv2
                cv2.setNumThreads(0)
            except Exception:
                pass
            torch.set_num_threads(1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

        if self.is_main_process:
            print(f"✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
            print(f"✓ Val: {len(val_dataset)} samples, {len(val_loader)} batches")

        return train_loader, val_loader


    def _build_optimizer(self, params_override=None):
        """Build optimizer"""
        if params_override is not None:
            params = params_override
        else:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=tuple(self.config['optimizer']['betas']),
            eps=self.config['optimizer']['eps']
        )

        return optimizer

    def _build_scheduler(self, optimizer_override=None):
        """Build learning rate scheduler"""
        optimizer = optimizer_override or self.optimizer
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        warmup_steps = self.config['training'].get('warmup_steps')
        warmup_ratio = self.config['training'].get('warmup_ratio')
        min_lr = self.config['training'].get('min_lr', None)
        scheduler_type = self.config['training'].get('lr_scheduler_type', 'cosine')

        if warmup_steps is None and warmup_ratio is not None:
            warmup_steps = int(total_steps * float(warmup_ratio))
        if warmup_steps is None:
            warmup_steps = 0

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            else:
                if total_steps <= warmup_steps:
                    return 1.0
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                if scheduler_type == 'cosine':
                    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
                    if min_lr is None:
                        return cosine
                    min_ratio = min_lr / self.config['training']['learning_rate']
                    return min_ratio + (1.0 - min_ratio) * cosine
                if scheduler_type == 'linear':
                    linear = max(0.0, 1.0 - progress)
                    if min_lr is None:
                        return linear
                    min_ratio = min_lr / self.config['training']['learning_rate']
                    return min_ratio + (1.0 - min_ratio) * linear
                if scheduler_type == 'constant':
                    return 1.0
                if scheduler_type == 'constant_with_warmup':
                    return 1.0
                raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return scheduler

    def _maybe_resume(self):
        resume_path = self.config['training'].get('resume_from')
        if not resume_path:
            return

        if self.use_deepspeed:
            load_dir = resume_path
            tag = self.config['training'].get('resume_tag')
            success, client_state = self.model.load_checkpoint(load_dir, tag=tag)
            if not success:
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir}")
            client_state = client_state or {}
            saved_epoch = client_state.get('epoch', 0)
            saved_step_in_epoch = client_state.get('step_in_epoch', 0)
            self.global_step = client_state.get('global_step', 0)
            self.best_val_acc = client_state.get('best_val_acc', 0.0)
            rng_state = client_state.get('rng_state')
            if rng_state:
                self._set_rng_state(rng_state)
            self._set_resume_position(saved_epoch, saved_step_in_epoch)
            if self.is_main_process:
                print(f"✓ Resumed DeepSpeed checkpoint from {load_dir} (epoch={self.start_epoch}, step={self.global_step}, skip={self.resume_step_in_epoch})")
        else:
            ckpt = torch.load(resume_path, map_location=f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            model_ref.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            saved_epoch = ckpt.get('epoch', 0)
            saved_step_in_epoch = ckpt.get('step_in_epoch', 0)
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_acc = ckpt.get('best_val_acc', 0.0)
            rng_state = ckpt.get('rng_state')
            if rng_state:
                self._set_rng_state(rng_state)
            self._set_resume_position(saved_epoch, saved_step_in_epoch)
            if self.is_main_process:
                print(f"✓ Resumed checkpoint from {resume_path} (epoch={self.start_epoch}, step={self.global_step}, skip={self.resume_step_in_epoch})")

    def _set_resume_position(self, saved_epoch, saved_step_in_epoch):
        if saved_step_in_epoch is None:
            saved_step_in_epoch = 0
        if saved_step_in_epoch >= len(self.train_loader):
            self.start_epoch = saved_epoch + 1
            self.resume_step_in_epoch = 0
        else:
            self.start_epoch = saved_epoch
            self.resume_step_in_epoch = saved_step_in_epoch

    def _get_rng_state(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state


    def _set_rng_state(self, state):
        try:
            random.setstate(state.get("python"))
            np.random.set_state(state.get("numpy"))
            torch.set_rng_state(state.get("torch"))
            if torch.cuda.is_available() and "cuda" in state:
                torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as e:
            if self.is_main_process:
                print(f"⚠️ Failed to restore RNG state: {e}")

    def _init_reporters(self):
        self.wandb_run = None
        self.mlflow_run = None
        if not self.is_main_process:
            return

        if 'wandb' in self.report_to:
            try:
                import wandb
                project = self.config['training'].get('wandb_project', 'CrossViewer')
                name = self.config['training'].get('run_name') or self.config['training'].get('wandb_name')
                self.wandb_run = wandb.init(project=project, name=name, config=self.config)
                print("✓ Weights & Biases logging enabled")
            except Exception as e:
                print(f"⚠️ Failed to init wandb: {e}")

        if 'mlflow' in self.report_to:
            try:
                import mlflow
                tracking_uri = self.config['training'].get('mlflow_tracking_uri')
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                run_name = self.config['training'].get('run_name')
                self.mlflow_run = mlflow.start_run(run_name=run_name)
                print("✓ MLflow logging enabled")
            except Exception as e:
                print(f"⚠️ Failed to init mlflow: {e}")

    def _log_metrics(self, metrics, step, split):
        if not self.is_main_process:
            return
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f'{split}/{k}', v, step)
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log({f'{split}/{k}': v for k, v in metrics.items()}, step=step)
            except Exception:
                pass
        if self.mlflow_run is not None:
            try:
                import mlflow
                for k, v in metrics.items():
                    mlflow.log_metric(f'{split}/{k}', v, step=step)
            except Exception:
                pass

    def _finalize_reporters(self):
        if not self.is_main_process:
            return
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if self.mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()

        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)

        loader_iter = iter(self.train_loader)

        pbar = tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch}",
            disable=not self.is_main_process,
        )

        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_vqa = 0.0
        epoch_acc = 0.0
        epoch_vqa_acc = 0.0

        timing_every = int(self.config['training'].get('timing_log_every', 0))
        timing_steps = int(self.config['training'].get('timing_log_steps', 0))
        debug_wait_every = bool(self.config['training'].get('debug_wait_every_batch', False))
        debug_interval = float(self.config['training'].get('debug_wait_interval', 10))

        for batch_idx in range(len(self.train_loader)):
            wait_start = time.time()
            if debug_wait_every and self.is_main_process:
                evt = threading.Event()

                def _heartbeat():
                    while not evt.wait(debug_interval):
                        elapsed = time.time() - wait_start
                        print(f"[debug] waiting batch {batch_idx}... {elapsed:.1f}s", flush=True)

                th = threading.Thread(target=_heartbeat, daemon=True)
                th.start()
            try:
                batch = next(loader_iter)
            except StopIteration:
                if debug_wait_every and self.is_main_process:
                    evt.set()
                break
            finally:
                if debug_wait_every and self.is_main_process:
                    evt.set()
            wait_time = time.time() - wait_start
            if debug_wait_every and self.is_main_process:
                print(f"[debug] batch {batch_idx} fetched in {wait_time:.3f}s", flush=True)
            if self.epoch == self.start_epoch and self.resume_step_in_epoch > 0 and batch_idx < self.resume_step_in_epoch:
                if self.is_main_process:
                    pbar.update(1)
                continue
            fwd_start = time.time()
            compute_infonce = self._resolve_compute_infonce(batch)
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                masks=batch['masks'],
                raw_images=batch.get('raw_images'),
                questions=batch['questions'],
                answers=batch['answers'],
                compute_infonce=compute_infonce,
                cutoff_len=self.config['data'].get('cutoff_len'),
                use_all_views_for_infonce=self.config['model'].get('use_all_views_for_infonce', False),
                use_all_views_for_vqa=self.config['model'].get('use_all_views_for_vqa', False),
                target_indices=batch.get('target_indices'),
                object_ids=batch.get('object_ids'),
                region_refs=batch.get('region_refs'),
                region_token_counts=batch.get('region_token_counts'),
                match_mode=self.config['model'].get('match_mode', 'gt'),
                region_source=self.config['model'].get('region_source', 'fused'),
                additional_pixel_values=batch.get('additional_pixel_values'),
                additional_grid_thw=batch.get('additional_grid_thw'),
                additional_masks=batch.get('additional_masks'),
                additional_box_params=batch.get('additional_box_params')
            )
            loss = outputs['loss']
            if self.is_main_process and torch.is_tensor(loss) and not torch.isfinite(loss).all():
                nan_keys = []

                def _check_nan(name, tensor):
                    if tensor is None or not torch.is_tensor(tensor):
                        return
                    if not torch.isfinite(tensor).all():
                        nan_keys.append(name)

                _check_nan("loss", loss)
                _check_nan("infonce_loss", outputs.get("infonce_loss"))
                _check_nan("vqa_loss", outputs.get("vqa_loss"))
                _check_nan("consistency_loss", outputs.get("consistency_loss"))
                _check_nan("answer_logits", outputs.get("answer_logits"))
                _check_nan("valid_mask", outputs.get("valid_mask"))
                emb = outputs.get("embeddings") or {}
                _check_nan("embeddings.ego", emb.get("ego"))
                _check_nan("embeddings.exo", emb.get("exo"))
                _check_nan("embeddings.fused", emb.get("fused"))
                _check_nan("embeddings.fused_objects", emb.get("fused_objects"))
                meta = {}
                if 'metadata' in batch and len(batch['metadata']) > 0:
                    meta = batch['metadata'][0] or {}
                meta_brief = {k: meta.get(k) for k in ("sample_id", "take_name", "frame_id", "question_type")}
                print(
                    f"[nan] step={self.global_step} batch={batch_idx} keys={nan_keys} "
                    f"loss={float(loss.detach().cpu()) if torch.is_tensor(loss) else loss} "
                    f"infonce={float(outputs['infonce_loss'].detach().cpu()) if outputs.get('infonce_loss') is not None else None} "
                    f"vqa={float(outputs['vqa_loss'].detach().cpu()) if outputs.get('vqa_loss') is not None else None} "
                    f"meta={meta_brief}",
                    flush=True
                )
            fwd_time = time.time() - fwd_start

            did_step = False
            bwd_start = time.time()
            if self.use_deepspeed:
                self.model.backward(loss)
                self.model.step()
                if hasattr(self.model, "is_gradient_accumulation_boundary"):
                    did_step = self.model.is_gradient_accumulation_boundary()
                else:
                    did_step = True
            else:
                loss = loss / max(1, self.grad_accum_steps)
                loss.backward()
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    did_step = True
            bwd_time = time.time() - bwd_start
            step_time = wait_time + fwd_time + bwd_time

            epoch_loss += loss.item()
            if outputs['infonce_loss'] is not None:
                epoch_infonce += outputs['infonce_loss'].item()
            if outputs['vqa_loss'] is not None:
                epoch_vqa += outputs['vqa_loss'].item()
            if outputs.get('infonce_acc') is not None:
                epoch_acc += float(outputs['infonce_acc'])
            if outputs.get('vqa_acc') is not None:
                epoch_vqa_acc += float(outputs['vqa_acc'])

            if self.is_main_process:
                debug_steps = int(self.config['training'].get('debug_steps', 0))
                if debug_steps > 0 and batch_idx < debug_steps:
                    infonce_val = outputs['infonce_loss'].item() if outputs['infonce_loss'] is not None else None
                    num_views = []
                    grid_list = batch.get('image_grid_thw') or []
                    raw_list = batch.get('raw_images') or []
                    timing_list = batch.get('timing') or []
                    for i in range(len(grid_list)):
                        g = grid_list[i]
                        if g is not None:
                            try:
                                num_views.append(int(g.shape[0]))
                                continue
                            except Exception:
                                pass
                        if i < len(raw_list) and raw_list[i] is not None:
                            try:
                                num_views.append(len(raw_list[i]))
                                continue
                            except Exception:
                                pass
                        if i < len(timing_list) and timing_list[i]:
                            num_views.append(timing_list[i].get('num_views'))
                        else:
                            num_views.append(None)
                    num_objects = []
                    if batch.get('masks') is not None:
                        for sample_masks in batch['masks']:
                            if len(sample_masks) == 0:
                                num_objects.append(0)
                            else:
                                try:
                                    num_objects.append(int(sample_masks[0].shape[0]))
                                except Exception:
                                    num_objects.append(-1)
                    compute_flags = compute_infonce
                    n_valid = outputs.get('debug_n_valid')
                    meta = {}
                    if 'metadata' in batch and len(batch['metadata']) > 0:
                        meta = batch['metadata'][0] or {}
                    meta_brief = {k: meta.get(k) for k in ("sample_id", "take_name", "frame_id", "num_objects", "question_type")}
                    msg = (f"[debug] batch={batch_idx} step={self.global_step} "
                           f"infonce={infonce_val} vqa={outputs['vqa_loss'].item() if outputs['vqa_loss'] is not None else None} "
                           f"views={num_views} objects={num_objects} compute_infonce={compute_flags} n_valid={n_valid} meta={meta_brief}")
                    try:
                        from tqdm import tqdm as _tqdm
                        _tqdm.write(msg)
                    except Exception:
                        print(msg, flush=True)

            if self.is_main_process and timing_every > 0:
                if timing_steps <= 0 or batch_idx < timing_steps:
                    if batch_idx % timing_every == 0:
                        timing = batch.get("timing") or []
                        if timing:
                            valid = [t for t in timing if isinstance(t, dict)]
                            if valid:
                                def _avg(key):
                                    return sum(t.get(key, 0.0) for t in valid) / len(valid)
                                load_avg = _avg("load_views")
                                proc_avg = _avg("processor")
                                add_avg = _avg("additional")
                                total_avg = _avg("total")
                                data_extra = f" load={load_avg:.3f}s proc={proc_avg:.3f}s crop={add_avg:.3f}s total={total_avg:.3f}s"
                            else:
                                data_extra = ""
                        else:
                            data_extra = ""
                        print(
                            f"[step timing] step={batch_idx} wait={wait_time:.3f}s fwd={fwd_time:.3f}s bwd={bwd_time:.3f}s total={step_time:.3f}s{data_extra}",
                            flush=True
                        )

            if self.is_main_process and batch_idx % self.config['training']['log_freq'] == 0:
                metrics = {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]}
                if outputs['infonce_loss'] is not None:
                    metrics['infonce_loss'] = outputs['infonce_loss'].item()
                if outputs['vqa_loss'] is not None:
                    metrics['vqa_loss'] = outputs['vqa_loss'].item()
                if outputs.get('infonce_acc') is not None:
                    metrics['infonce_acc'] = float(outputs['infonce_acc'])
                if outputs.get('vqa_acc') is not None:
                    metrics['vqa_acc'] = float(outputs['vqa_acc'])
                self._log_metrics(metrics, self.global_step, 'train')

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'i_loss': f"{outputs['infonce_loss'].item():.4f}" if outputs['infonce_loss'] is not None else "0.0000",
                    'vqa': f"{outputs['vqa_loss'].item():.4f}" if outputs['vqa_loss'] is not None else "0.0000",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

            if self.is_main_process:
                pbar.update(1)

            if did_step:
                self.global_step += 1

            eval_steps = self.config['training'].get('eval_steps')
            if did_step and eval_steps and self.global_step % eval_steps == 0:
                self.evaluate()
                self.model.train()

            save_steps = self.config['training'].get('save_steps')
            if did_step and save_steps and self.global_step % save_steps == 0:
                if self.use_deepspeed:
                    self.save_checkpoint(tag=f"step{self.global_step}", step_in_epoch=batch_idx + 1)
                else:
                    self.save_checkpoint(filename=f"checkpoint_step{self.global_step}.pth", step_in_epoch=batch_idx + 1)

        if self.is_main_process:
            pbar.close()

        avg_loss = epoch_loss / len(self.train_loader)
        avg_infonce = epoch_infonce / len(self.train_loader)
        avg_vqa = epoch_vqa / len(self.train_loader)
        avg_acc = epoch_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        avg_vqa_acc = epoch_vqa_acc / len(self.train_loader) if len(self.train_loader) > 0 else 0.0

        return avg_loss, avg_infonce, avg_vqa, avg_acc, avg_vqa_acc

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()

        total_loss = 0.0
        total_infonce = 0.0
        total_vqa = 0.0
        total_acc = 0.0
        total_vqa_acc = 0.0

        pbar = tqdm(self.val_loader, desc="Evaluating", disable=not self.is_main_process)

        for batch in pbar:
            outputs = self.model(
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                masks=batch['masks'],
                raw_images=batch.get('raw_images'),
                questions=batch['questions'],
                answers=batch['answers'],
                compute_infonce=self._resolve_compute_infonce(batch),
                cutoff_len=self.config['data'].get('cutoff_len'),
                use_all_views_for_infonce=self.config['model'].get('use_all_views_for_infonce', False),
                use_all_views_for_vqa=self.config['model'].get('use_all_views_for_vqa', False),
                target_indices=batch.get('target_indices'),
                object_ids=batch.get('object_ids'),
                region_refs=batch.get('region_refs'),
                region_token_counts=batch.get('region_token_counts'),
                match_mode=self.config['model'].get('match_mode', 'gt'),
                region_source=self.config['model'].get('region_source', 'fused'),
                additional_pixel_values=batch.get('additional_pixel_values'),
                additional_grid_thw=batch.get('additional_grid_thw'),
                additional_masks=batch.get('additional_masks'),
                additional_box_params=batch.get('additional_box_params')
            )

            total_loss += outputs['loss'].item()
            if outputs['infonce_loss'] is not None:
                total_infonce += outputs['infonce_loss'].item()
            if outputs['vqa_loss'] is not None:
                total_vqa += outputs['vqa_loss'].item()
            if outputs.get('infonce_acc') is not None:
                total_acc += float(outputs['infonce_acc'])
            if outputs.get('vqa_acc') is not None:
                total_vqa_acc += float(outputs['vqa_acc'])

        avg_loss = total_loss / len(self.val_loader)
        avg_infonce = total_infonce / len(self.val_loader)
        avg_vqa = total_vqa / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        avg_vqa_acc = total_vqa_acc / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

        if self.is_main_process:
            print(f"\nValidation - Loss: {avg_loss:.4f}, InfoNCE: {avg_infonce:.4f}, VQA: {avg_vqa:.4f}, InfoNCE Acc: {avg_acc:.4f}, VQA Acc: {avg_vqa_acc:.4f}")
            self._log_metrics({
                'loss': avg_loss,
                'infonce_loss': avg_infonce,
                'vqa_loss': avg_vqa,
                'infonce_acc': avg_acc,
                'vqa_acc': avg_vqa_acc
            }, self.epoch, 'val')

        return avg_loss, avg_infonce, avg_vqa, avg_acc, avg_vqa_acc

    def save_checkpoint(self, filename="checkpoint.pth", tag=None, step_in_epoch=None):
        """Save checkpoint"""
        save_dir = Path(self.config['training']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_only_model = bool(self.config['training'].get('save_only_model', False))

        if self.use_deepspeed:
            tag = tag or (filename if filename else f"epoch{self.epoch}")
            if tag.endswith(".pth"):
                tag = tag[:-4]
            if save_only_model:
                if self.is_main_process:
                    model_ref = self.model.module if hasattr(self.model, 'module') else self.model
                    save_path = save_dir / f"only_model_{tag}.pth"
                    torch.save(model_ref.state_dict(), save_path)
                    print(f"✓ Saved model-only weights to {save_path}")
            else:
                client_state = {
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'best_val_acc': self.best_val_acc,
                    'step_in_epoch': step_in_epoch,
                    'rng_state': self._get_rng_state()
                }
                self.model.save_checkpoint(str(save_dir), tag=tag, client_state=client_state)
                if self.is_main_process:
                    print(f"✓ DeepSpeed checkpoint saved to {save_dir} (tag={tag})")
        else:
            if not self.is_main_process:
                return
            model = self.model.module if hasattr(self.model, 'module') else self.model
            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'best_val_acc': self.best_val_acc,
                'step_in_epoch': step_in_epoch,
                'rng_state': self._get_rng_state(),
                'config': self.config
            }
            if save_only_model:
                checkpoint['model_state_dict'] = model.state_dict()
            else:
                checkpoint['model_state_dict'] = model.state_dict()
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            save_path = save_dir / filename
            torch.save(checkpoint, save_path)
            print(f"✓ Checkpoint saved to {save_path}")

    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            self.epoch = epoch

            train_loss, train_infonce, train_vqa, train_acc, train_vqa_acc = self.train_epoch()

            if self.is_main_process:
                print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, InfoNCE: {train_infonce:.4f}, VQA: {train_vqa:.4f}, InfoNCE Acc: {train_acc:.4f}, VQA Acc: {train_vqa_acc:.4f}")
                self.loss_history.append({'train': train_loss})

            if epoch % self.config['training']['eval_freq'] == 0:
                val_loss, val_infonce, val_vqa, val_acc, val_vqa_acc = self.evaluate()

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint("best.pth", step_in_epoch=len(self.train_loader))
                if self.is_main_process:
                    self.loss_history.append({'val': val_loss})

            save_steps = self.config['training'].get('save_steps')
            if not save_steps and epoch % self.config['training']['save_freq'] == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch}.pth", step_in_epoch=len(self.train_loader))

        if self.is_main_process:
            print(f"\n✓ Training complete! Best val acc: {self.best_val_acc:.4f}")
            self.writer.close()
            if self.config['training'].get('plot_loss'):
                self._plot_loss_curve()
            self._finalize_reporters()

    def _plot_loss_curve(self):
        if not self.is_main_process:
            return
        log_dir = self.writer.log_dir if self.writer else self.config['training']['log_dir']
        try:
            import matplotlib.pyplot as plt
            train_losses = [x['train'] for x in self.loss_history if 'train' in x]
            val_losses = [x['val'] for x in self.loss_history if 'val' in x]
            steps = list(range(len(train_losses)))
            plt.figure(figsize=(6, 4))
            plt.plot(steps, train_losses, label='train')
            if val_losses:
                plt.plot(range(len(val_losses)), val_losses, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            out_path = os.path.join(log_dir, 'loss_curve.png')
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"✓ Saved loss curve to {out_path}")
        except Exception as e:
            out_path = os.path.join(log_dir, 'loss_history.json')
            with open(out_path, 'w') as f:
                json.dump(self.loss_history, f, ensure_ascii=False, indent=2)
            print(f"⚠️ Plot failed ({e}); saved loss history to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_required_paths(
        config,
        (
            ("model", "vision_encoder_path"),
            ("data", "data_root"),
            ("data", "jsonl_train"),
            ("data", "jsonl_val"),
        ),
    )

    use_deepspeed = config['training'].get('use_deepspeed', False)
    rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    env_world_size = os.environ.get("WORLD_SIZE")
    if use_deepspeed:
        world_size = int(env_world_size) if env_world_size is not None else int(config['training'].get('num_gpus', 1))
    else:
        if config['training']['use_ddp']:
            world_size = int(env_world_size) if env_world_size is not None else int(config['training'].get('num_gpus', 1))
        else:
            world_size = 1

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if rank != 0 and config['training'].get('rank0_only_log', True):
        sys.stdout = open(os.devnull, 'w')

    if use_deepspeed:
        try:
            import deepspeed
        except ImportError:
            raise RuntimeError("DeepSpeed is enabled in config but not installed.")
        deepspeed.init_distributed()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    elif world_size > 1:
        setup_ddp(rank, world_size)

    ds_config = config['training'].get('deepspeed_config')
    if use_deepspeed and not ds_config:
        raise RuntimeError("DeepSpeed is enabled but training.deepspeed_config is not set.")
    trainer = Trainer(config, rank=rank, world_size=world_size, use_deepspeed=use_deepspeed, deepspeed_config=ds_config)
    trainer.train()

    if world_size > 1 and not use_deepspeed:
        cleanup_ddp()


if __name__ == "__main__":
    main()
