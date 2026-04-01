#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


OPTION_RE = re.compile(r"^([A-Z])\.\s*(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple-choice accuracy on validation set.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (.pth) or DeepSpeed folder.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for fast eval.")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save per-type metrics JSON.")
    parser.add_argument(
        "--debug-qtypes",
        type=str,
        default=None,
        help="Comma-separated question types to dump detailed predictions, e.g. Q6,Q8,Q14.",
    )
    parser.add_argument(
        "--only-qtypes",
        type=str,
        default=None,
        help="Only evaluate these question types (comma-separated), e.g. Q6,Q8,Q14.",
    )
    parser.add_argument(
        "--debug-max-per-type",
        type=int,
        default=50,
        help="Max debug samples per question type (per run).",
    )
    parser.add_argument(
        "--debug-out",
        type=str,
        default=None,
        help="Write debug samples to a JSONL file.",
    )
    return parser.parse_args()


def extract_options(question: str) -> List[Tuple[str, str]]:
    options = []
    for line in question.splitlines():
        m = OPTION_RE.match(line.strip())
        if m:
            label = m.group(1).upper()
            text = m.group(2).strip()
            options.append((label, text))
    seen = set()
    uniq = []
    for label, text in options:
        if label not in seen:
            seen.add(label)
            uniq.append((label, text))
    return uniq


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    text = str(ans).strip()
    m = re.match(r"([A-Za-z])", text)
    if not m:
        return None
    return m.group(1).upper()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def answer_to_label(answer: Optional[str], options: List[Tuple[str, str]]) -> Optional[str]:
    """
    Map an answer string to an option label.
    Accepts answers like "A" or "B", or answer text/number that matches option text.
    """
    if answer is None:
        return None
    ans = str(answer).strip()
    if not options:
        return None

    # If answer is already a letter, use it directly.
    m = re.match(r"^([A-Za-z])", ans)
    if m:
        return m.group(1).upper()

    ans_norm = normalize_text(ans)
    # Try to match answer text against option text.
    matches = []
    for label, opt_text in options:
        opt_norm = normalize_text(opt_text)
        if ans_norm == opt_norm:
            return label
        if ans_norm and (ans_norm in opt_norm or opt_norm in ans_norm):
            matches.append(label)
    if len(matches) == 1:
        return matches[0]

    # If answer is a digit and options are like "A. 3", map by prefix or index.
    if ans.isdigit():
        for label, opt_text in options:
            opt_norm = normalize_text(opt_text)
            if opt_norm.startswith(ans):
                return label
        idx = int(ans) - 1
        if 0 <= idx < len(options):
            return options[idx][0]

    return None


def label_token_ids(tokenizer, label: str) -> List[int]:
    variants = [
        f" {label}",
        label,
        f"{label}.",
        f" {label}.",
        f"{label})",
        f" {label})",
        f"{label}、",
        f" {label}、",
    ]
    ids = []
    for cand in variants:
        tok = tokenizer.encode(cand, add_special_tokens=False)
        if len(tok) == 1:
            ids.append(tok[0])
    # unique preserve order
    seen = set()
    uniq = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            uniq.append(tid)
    return uniq


def parse_qtypes(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    items = []
    for part in str(raw).split(","):
        item = part.strip()
        if item:
            items.append(item)
    return items


def build_qtype_indices(jsonl_path: str, qtypes: List[str], max_samples: Optional[int]) -> List[int]:
    qset = {q.strip().upper() for q in qtypes if q.strip()}
    if not qset:
        return []
    indices: List[int] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            qtype = str(obj.get("question_type") or "").upper()
            if qtype in qset:
                indices.append(i)
    return indices


def build_val_loader(
    config: Dict,
    batch_size: Optional[int],
    num_workers: Optional[int],
    max_samples: Optional[int],
    distributed: bool,
    rank: int,
    world_size: int,
    only_qtypes: Optional[List[str]] = None,
) -> DataLoader:
    cfg = config["data"]
    val_dataset = CrossViewerJSONLDataset(
        jsonl_path=cfg["jsonl_val"],
        data_root=cfg["data_root"],
        processor_path=config["model"]["vision_encoder_path"],
        split="val",
        max_samples=max_samples,
        lazy_load=cfg.get("lazy_load", True),
        index_cache=cfg.get("index_cache", True),
        include_options_in_question=cfg.get("include_options_in_question", True),
        load_masks=cfg.get("load_masks", True),
        prompt_template=cfg.get("prompt_template"),
        max_retries=cfg.get("max_retries", 20),
        load_timeout_sec=cfg.get("load_timeout_sec", 30),
        video_backend=cfg.get("video_backend", "decord"),
        decord_num_threads=cfg.get("decord_num_threads", 1),
        mask_jitter=False,
        mask_jitter_prob=cfg.get("mask_jitter_prob", 0.3),
        mask_jitter_max_shift=cfg.get("mask_jitter_max_shift", 10),
        mask_jitter_max_kernel=cfg.get("mask_jitter_max_kernel", 5),
        mask_num=cfg.get("mask_num", 32),
        region_token_num=cfg.get("region_token_num", None),
        region_placeholder=cfg.get("region_placeholder", "<region>"),
        resample_strategy=cfg.get("resample_strategy", "random"),
        image_only=cfg.get("image_only", False),
        use_additional_inputs=cfg.get("use_additional_inputs", False),
        use_processor=cfg.get("use_processor", True),
        return_raw_images=cfg.get("return_raw_images", False),
        defer_region_token_expansion=cfg.get("defer_region_token_expansion", False),
        debug_timing=False,
        timing_every=0,
        timing_max_logs=0,
        debug_trace=False,
        debug_trace_max=0,
        debug_trace_every=0,
        max_objects_per_sample=cfg.get("max_objects_per_sample", -1),
    )
    if only_qtypes:
        indices = build_qtype_indices(cfg["jsonl_val"], only_qtypes, max_samples)
        if rank == 0:
            print(f"[eval] filtering qtypes={only_qtypes}, kept={len(indices)} samples", flush=True)
        val_dataset = Subset(val_dataset, indices)

    bs = batch_size or config["evaluation"]["batch_size"]
    nw = num_workers if num_workers is not None else int(cfg.get("num_workers", 0))
    sampler = None
    if distributed:
        sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False if sampler is None else False,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=bool(cfg.get("pin_memory", True)),
        prefetch_factor=cfg.get("prefetch_factor", 2) if nw > 0 else None,
        persistent_workers=bool(cfg.get("persistent_workers", False)) if nw > 0 else False,
    )
    return loader


def load_checkpoint(model: torch.nn.Module, ckpt_path: Optional[str], verbose: bool = True) -> None:
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if path.is_dir():
        mp_rank_path = path / "mp_rank_00_model_states.pt"
        if mp_rank_path.exists():
            if verbose:
                print(f"[ckpt] loading mp_rank_00_model_states.pt from {mp_rank_path} ...", flush=True)
            try:
                obj = torch.load(str(mp_rank_path), map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(str(mp_rank_path), map_location="cpu")
            if isinstance(obj, dict):
                if "module" in obj:
                    state_dict = obj["module"]
                elif "model" in obj:
                    state_dict = obj["model"]
                else:
                    state_dict = obj
            else:
                state_dict = obj
            if verbose:
                print("[ckpt] loaded mp_rank_00_model_states.pt", flush=True)
        else:
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            except Exception as exc:
                raise RuntimeError("DeepSpeed is required to load a ZeRO checkpoint directory.") from exc
            if verbose:
                print(f"[ckpt] loading ZeRO checkpoint from {path} ...", flush=True)
            state_dict = get_fp32_state_dict_from_zero_checkpoint(str(path))
            if verbose:
                print("[ckpt] loaded ZeRO checkpoint", flush=True)
    else:
        if verbose:
            print(f"[ckpt] loading {path} ...", flush=True)
        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(path), map_location="cpu")
        state_dict = obj.get("model_state_dict", obj)
        if verbose:
            print("[ckpt] loaded .pth checkpoint", flush=True)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(
            f"Loaded checkpoint: missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )


def setup_distributed(device: torch.device) -> Tuple[bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_required_paths(
        config,
        (
            ("model", "vision_encoder_path"),
            ("data", "data_root"),
            ("data", "jsonl_val"),
        ),
    )
    debug_qtypes = set(parse_qtypes(args.debug_qtypes))
    only_qtypes = set(parse_qtypes(args.only_qtypes))
    debug_max = int(args.debug_max_per_type or 0)
    debug_counts: Dict[str, int] = {q: 0 for q in debug_qtypes}
    debug_samples: List[Dict] = []

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1 and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and config["model"].get("preprocess_on_gpu", False):
        raise RuntimeError("preprocess_on_gpu is true but device is CPU. Use a CUDA device or disable it.")

    distributed, rank, world_size = setup_distributed(device)

    model_cls = _get_model_class(config)
    model = model_cls(
        vision_encoder_path=config["model"]["vision_encoder_path"],
        freeze_vision_encoder=config["model"]["freeze_vision_encoder"],
        freeze_llm=config["model"].get("freeze_llm", True),
        num_object_tokens=config["model"]["num_object_tokens"],
        num_cross_attn_heads=config["model"]["num_cross_attn_heads"],
        contrast_dim=config["model"]["contrast_dim"],
        temperature=config["loss"]["temperature"],
        infonce_weight=config["loss"].get("info_nce_weight", 1.0),
        vqa_weight=config["loss"].get("vqa_weight", 0.5),
        triplet_weight=config["loss"].get("triplet_weight", 0.0),
        attn_implementation=config["model"].get("attn_implementation"),
        load_device=None,
        low_cpu_mem_usage=config["model"].get("low_cpu_mem_usage", True),
        match_mode=config["model"].get("match_mode", "gt"),
        region_source=config["model"].get("region_source", "fused"),
        pixelrefer_mode=config["model"].get("pixelrefer_mode", "full"),
        preprocess_on_gpu=config["model"].get("preprocess_on_gpu", False),
        region_placeholder=config["data"].get("region_placeholder", "<region>"),
        mask_num=config["data"].get("mask_num", 32),
        debug_pixelrefer=False,
        debug_pixelrefer_max_steps=0,
        debug_pixelrefer_max_masks=0,
        debug_pixelrefer_every=1,
        debug_pixelrefer_per_mask_preprocess=False,
        debug_match=False,
        debug_max_steps=0,
        debug_nan=False,
        freeze_pos_encoder=config["model"].get("freeze_pos_encoder", False),
        unfreeze_lm_head=config["model"].get("unfreeze_lm_head", False),
    )
    load_checkpoint(model, args.ckpt, verbose=(not distributed or rank == 0))
    model.to(device)
    model.eval()

    loader = build_val_loader(
        config,
        args.batch_size,
        args.num_workers,
        args.max_samples,
        distributed,
        rank,
        world_size,
        only_qtypes=list(only_qtypes) if only_qtypes else None,
    )
    tokenizer = model.tokenizer

    total = 0
    correct = 0
    skipped = 0
    per_type: Dict[str, List[int]] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", total=len(loader), disable=distributed and rank != 0):
            outputs = model(
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                masks=batch["masks"],
                raw_images=batch.get("raw_images"),
                questions=batch["questions"],
                answers=None,
                compute_infonce=[False] * len(batch["questions"]),
                cutoff_len=config["data"].get("cutoff_len"),
                use_all_views_for_vqa=config["model"].get("use_all_views_for_vqa", False),
                use_all_views_for_infonce=False,
                target_indices=batch.get("target_indices"),
                object_ids=batch.get("object_ids"),
                region_refs=batch.get("region_refs"),
                region_token_counts=batch.get("region_token_counts"),
                additional_pixel_values=batch.get("additional_pixel_values"),
                additional_grid_thw=batch.get("additional_grid_thw"),
                additional_masks=batch.get("additional_masks"),
                additional_box_params=batch.get("additional_box_params"),
            )

            logits = outputs["answer_logits"]  # [B, vocab]
            for i, question in enumerate(batch["questions"]):
                options = extract_options(question)
                labels = [lab for lab, _ in options]
                gt = answer_to_label(batch["answers"][i], options)
                qtype = (batch.get("metadata") or [{}])[i].get("question_type") or "UNK"
                if only_qtypes and qtype not in only_qtypes:
                    continue

                do_debug = qtype in debug_qtypes and debug_counts.get(qtype, 0) < debug_max
                debug_entry: Optional[Dict] = None
                if do_debug:
                    debug_entry = {
                        "qtype": qtype,
                        "sample_id": (batch.get("metadata") or [{}])[i].get("sample_id"),
                        "take_name": (batch.get("metadata") or [{}])[i].get("take_name"),
                        "frame_id": (batch.get("metadata") or [{}])[i].get("frame_id"),
                        "question": question,
                        "options": options,
                        "answer_raw": batch["answers"][i],
                        "gt": gt,
                    }

                if not labels or gt is None:
                    if debug_entry is not None:
                        debug_entry["skipped_reason"] = "missing_labels_or_gt"
                        debug_samples.append(debug_entry)
                        debug_counts[qtype] = debug_counts.get(qtype, 0) + 1
                    skipped += 1
                    continue
                token_ids_per_label = []
                token_id_map: Dict[str, List[int]] = {}
                for lab in labels:
                    tids = label_token_ids(tokenizer, lab)
                    if not tids:
                        token_ids_per_label = []
                        break
                    token_ids_per_label.append(tids)
                    token_id_map[lab] = tids
                if not token_ids_per_label:
                    if debug_entry is not None:
                        debug_entry["skipped_reason"] = "no_label_token_ids"
                        debug_entry["label_token_ids"] = token_id_map
                        debug_samples.append(debug_entry)
                        debug_counts[qtype] = debug_counts.get(qtype, 0) + 1
                    skipped += 1
                    continue

                scores = []
                score_map: Dict[str, float] = {}
                for tids in token_ids_per_label:
                    scores.append(torch.max(logits[i, tids]))
                scores = torch.stack(scores, dim=0)
                pred_idx = int(torch.argmax(scores).item())
                pred_label = labels[pred_idx]
                for lab, tids in zip(labels, token_ids_per_label):
                    score_map[lab] = float(torch.max(logits[i, tids]).item())

                if debug_entry is not None:
                    debug_entry.update(
                        {
                            "pred": pred_label,
                            "scores": score_map,
                            "label_token_ids": token_id_map,
                        }
                    )
                    debug_samples.append(debug_entry)
                    debug_counts[qtype] = debug_counts.get(qtype, 0) + 1

                total += 1
                per_type.setdefault(qtype, [0, 0])[1] += 1
                if pred_label == gt:
                    correct += 1
                    per_type[qtype][0] += 1

    if distributed:
        counts = torch.tensor([correct, total, skipped], dtype=torch.long, device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        correct, total, skipped = counts.tolist()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, per_type)
        if rank == 0:
            merged: Dict[str, List[int]] = {}
            for d in gathered:
                if not d:
                    continue
                for k, v in d.items():
                    merged.setdefault(k, [0, 0])
                    merged[k][0] += int(v[0])
                    merged[k][1] += int(v[1])
            per_type = merged
        if debug_qtypes:
            gathered_dbg = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_dbg, debug_samples)
            if rank == 0:
                merged_dbg: List[Dict] = []
                for d in gathered_dbg:
                    if d:
                        merged_dbg.extend(d)
                debug_samples = merged_dbg

    if not distributed or rank == 0:
        overall = (correct / total) if total > 0 else 0.0
        print(f"\nOverall MC accuracy: {overall:.4f} ({correct}/{total}), skipped={skipped}")
        print("Per-question-type accuracy:")
        for qtype in sorted(per_type.keys()):
            c, t = per_type[qtype]
            acc = (c / t) if t > 0 else 0.0
            print(f"  {qtype}: {acc:.4f} ({c}/{t})")

        if args.save_json:
            out = {
                "overall": {"acc": overall, "correct": correct, "total": total, "skipped": skipped},
                "per_type": {k: {"acc": (v[0] / v[1]) if v[1] > 0 else 0.0, "correct": v[0], "total": v[1]}
                             for k, v in per_type.items()},
            }
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"Saved metrics to {args.save_json}")

        if args.debug_out and debug_qtypes:
            # Enforce a global cap per type in case of multi-rank merge.
            cap = debug_max if debug_max > 0 else None
            counts: Dict[str, int] = {}
            filtered: List[Dict] = []
            for item in debug_samples:
                qtype = item.get("qtype", "UNK")
                if qtype not in debug_qtypes:
                    continue
                if cap is not None:
                    if counts.get(qtype, 0) >= cap:
                        continue
                filtered.append(item)
                counts[qtype] = counts.get(qtype, 0) + 1
            with open(args.debug_out, "w", encoding="utf-8") as f:
                for item in filtered:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved debug samples to {args.debug_out}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
