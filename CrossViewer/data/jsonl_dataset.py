"""
JSONL dataset loader for CrossViewer (combined samples).
Loads per-view images + masks directly from JSONL "views".
"""
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, get_worker_info

from pycocotools import mask as mask_utils
from .mask_utils import prepare_additional_inputs
from .object_utils import extract_object_category


def decode_mask(mask_dict):
    """Decode mask from COCO RLE dict."""
    if mask_dict is None:
        return None
    if isinstance(mask_dict, dict) and "counts" in mask_dict and "size" in mask_dict:
        rle_obj = dict(mask_dict)
        counts = rle_obj.get("counts")
        if isinstance(counts, list):
            rle_obj = mask_utils.frPyObjects(rle_obj, rle_obj["size"][0], rle_obj["size"][1])
        elif isinstance(counts, str):
            rle_obj["counts"] = counts.encode("utf-8")
        mask = mask_utils.decode(rle_obj)
    else:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _bbox_to_mask(bbox_xyxy, h, w):
    if bbox_xyxy is None:
        return np.zeros((h, w), dtype=np.float32)
    try:
        x1, y1, x2, y2 = bbox_xyxy
    except Exception:
        return np.zeros((h, w), dtype=np.float32)
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1 or y2 < y1:
        return np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2 + 1, x1:x2 + 1] = 1.0
    return mask


def _shift_mask(mask, dx, dy):
    if dx == 0 and dy == 0:
        return mask
    shifted = np.roll(mask, shift=(dy, dx), axis=(0, 1))
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted


def jitter_mask(mask, prob=0.3, max_shift=10, max_kernel=5):
    """
    Apply random morphological jitter to a binary mask.
    """
    if random.random() > prob:
        return mask

    mask_bin = (mask > 0.5).astype(np.uint8)
    op = random.choice(["erode", "dilate", "shift", "none"])
    if op == "erode":
        k = random.randint(1, max_kernel)
        kernel = np.ones((k, k), np.uint8)
        mask_bin = cv2.erode(mask_bin, kernel, iterations=1)
    elif op == "dilate":
        k = random.randint(1, max_kernel)
        kernel = np.ones((k, k), np.uint8)
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
    elif op == "shift":
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        mask_bin = _shift_mask(mask_bin, dx, dy)

    return mask_bin.astype(mask.dtype)


def infer_target_object_name(metadata, spatial_info):
    """
    Infer target object name for Re-ID questions when object_name is missing.
    Uses source_view + object_category + source_ordinal over spatial_info.
    """
    if not metadata or not spatial_info:
        return None

    source_view = metadata.get("source_view", None)
    object_category = metadata.get("object_category", None)
    source_ordinal = metadata.get("source_ordinal", None)
    if source_view is None or object_category is None:
        return None

    view_key = f"view_{source_view}"
    view_info = spatial_info.get(view_key, {})
    sorted_objects = view_info.get("sorted_objects", [])
    if not sorted_objects:
        return None

    same_category = [
        obj_name for obj_name in sorted_objects
        if extract_object_category(obj_name) == object_category
    ]
    if not same_category:
        return None

    ordinal_map = {
        "the first": 0,
        "the second": 1,
        "the third": 2,
        "the fourth": 3,
        "the fifth": 4,
        "the sixth": 5,
        "the seventh": 6,
    }
    if source_ordinal in (None, "the"):
        ordinal_idx = 0
    else:
        ordinal_idx = ordinal_map.get(source_ordinal, 0)

    if ordinal_idx >= len(same_category):
        ordinal_idx = 0

    return same_category[ordinal_idx]


class CrossViewerJSONLDataset(Dataset):
    """
    Dataset that loads combined JSONL samples and decodes masks from inline views.
    """

    def __init__(self,
                 jsonl_path,
                 data_root,
                 processor_path=None,
                 split="train",
                 max_samples=None,
                 lazy_load=True,
                 index_cache=True,
                 include_options_in_question=True,
                 load_masks=True,
                 prompt_template=None,
                 max_retries=20,
                 load_timeout_sec=30,
                 video_backend="decord",
                 decord_num_threads=1,
                 mask_jitter=False,
                 mask_jitter_prob=0.3,
                 mask_jitter_max_shift=10,
                 mask_jitter_max_kernel=5,
                 mask_num=32,
                 region_token_num=None,
                 region_placeholder="<region>",
                 resample_strategy="random",
                 image_only=False,
                 use_additional_inputs=True,
                 use_processor=True,
                 return_raw_images=False,
                 defer_region_token_expansion=False,
                 debug_timing=False,
                 timing_every=200,
                 timing_max_logs=50,
                 debug_trace=False,
                 debug_trace_max=20,
                 debug_trace_every=1,
                 max_objects_per_sample=-1):
        self.data_root = Path(data_root)
        self.processor_path = processor_path
        self.split = split
        self.include_options = include_options_in_question
        self.load_masks = load_masks
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.load_timeout_sec = load_timeout_sec
        self.video_backend = video_backend
        self.decord_num_threads = decord_num_threads
        self.mask_jitter = mask_jitter
        self.mask_jitter_prob = mask_jitter_prob
        self.mask_jitter_max_shift = mask_jitter_max_shift
        self.mask_jitter_max_kernel = mask_jitter_max_kernel
        self.mask_num = int(mask_num)
        self.region_token_num = int(region_token_num or self.mask_num)
        self.region_placeholder = region_placeholder
        self.region_token = "<REGION>"
        self.lazy_load = bool(lazy_load)
        self.index_cache = bool(index_cache)
        self.resample_strategy = str(resample_strategy or "random").lower()
        self.image_only = bool(image_only)
        self.use_additional_inputs = bool(use_additional_inputs)
        self.use_processor = bool(use_processor)
        self.return_raw_images = bool(return_raw_images)
        self.defer_region_token_expansion = bool(defer_region_token_expansion)
        self.debug_timing = bool(debug_timing)
        self.timing_every = int(timing_every)
        self.timing_max_logs = int(timing_max_logs)
        self._timing_logged = 0
        self.debug_trace = bool(debug_trace)
        self.debug_trace_max = int(debug_trace_max)
        self.debug_trace_every = int(debug_trace_every)
        self._trace_logged = 0
        try:
            self.max_objects_per_sample = int(max_objects_per_sample)
        except Exception:
            self.max_objects_per_sample = -1
        self.max_samples = max_samples
        self._fp = None
        self._offsets = None
        self._vr_cache = {}
        self.bad_samples = set()
        self.bad_videos = set()
        self.warned_samples = set()
        self.warned_videos = set()
        self._is_main_process = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0

        self.samples = None
        self.jsonl_path = Path(jsonl_path)
        if self.lazy_load:
            if self._is_main_process:
                print(f"Indexing {split} JSONL from {self.jsonl_path}...")
            self._offsets = self._load_or_build_offsets()
            if self.max_samples is not None:
                self._offsets = self._offsets[: self.max_samples]
            if self._is_main_process:
                print(f"✓ Indexed {len(self._offsets)} samples")
        else:
            self.samples = []
            if self._is_main_process:
                print(f"Loading {split} JSONL from {self.jsonl_path}...")
            with self.jsonl_path.open("r") as f:
                for i, line in enumerate(f):
                    if max_samples is not None and i >= max_samples:
                        break
                    self.samples.append(json.loads(line))
            if self._is_main_process:
                print(f"✓ Loaded {len(self.samples)} samples")

        self.processor = None
        if self.use_processor or self.use_additional_inputs:
            if not self.processor_path:
                raise ValueError(
                    "processor_path must be provided when use_processor or use_additional_inputs is enabled."
                )
            from transformers import AutoProcessor
            if self._is_main_process:
                print(f"Loading Qwen3-VL processor from {self.processor_path}...")
            self.processor = AutoProcessor.from_pretrained(
                self.processor_path,
                trust_remote_code=True
            )

    def __len__(self):
        if self.lazy_load:
            return len(self._offsets)
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def load_video_frame(self, video_path, frame_id):
        """Load a specific frame from video with optional timeout"""
        if self.video_backend == "decord":
            try:
                from decord import VideoReader, cpu
            except Exception as e:
                raise ImportError("decord is not installed. Please `pip install decord` in llm env.") from e

            vr = self._vr_cache.get(video_path)
            if vr is None:
                vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=self.decord_num_threads)
                if len(self._vr_cache) >= 8:
                    self._vr_cache.pop(next(iter(self._vr_cache)))
                self._vr_cache[video_path] = vr

            if frame_id >= len(vr):
                raise ValueError(f"frame_id {frame_id} >= num_frames {len(vr)} for {video_path}")
            frame = vr[frame_id].asnumpy()  # RGB
            return Image.fromarray(frame)

        def _read_frame():
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_id} from {video_path}")
            return frame

        if self.load_timeout_sec and self.load_timeout_sec > 0:
            import signal

            def _handle_timeout(signum, frame_):
                raise TimeoutError(f"Timeout reading frame {frame_id} from {video_path}")

            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, float(self.load_timeout_sec))
            try:
                frame = _read_frame()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            frame = _read_frame()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _resolve_image_path(self, image_path):
        if image_path is None:
            return None
        img_path = Path(image_path)
        if not img_path.is_absolute():
            primary = self.data_root / img_path
            if primary.exists():
                return primary
            alt_root = self.data_root.parent
            alt = alt_root / img_path if alt_root is not None else None
            if alt is not None and alt.exists():
                return alt
            img_path = primary
        return img_path

    def _load_image(self, image_path):
        img_path = self._resolve_image_path(image_path)
        if img_path is None:
            return None
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def _load_or_build_offsets(self):
        idx_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".idx.npy")
        meta_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".idx.json")
        lock_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".idx.lock")

        file_stat = self.jsonl_path.stat()
        file_size = int(file_stat.st_size)
        file_mtime = int(file_stat.st_mtime)

        def _try_load_cached():
            if not (idx_path.exists() and meta_path.exists()):
                return None
            try:
                with meta_path.open("r") as f:
                    meta = json.load(f)
                if meta.get("file_size") == file_size and meta.get("file_mtime") == file_mtime:
                    return np.load(idx_path, mmap_mode="r")
            except Exception:
                return None
            return None

        if self.index_cache:
            cached = _try_load_cached()
            if cached is not None:
                return cached

            got_lock = False
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                got_lock = True
            except FileExistsError:
                got_lock = False

            if not got_lock:
                wait_sec = 0.2
                for _ in range(600):  # up to ~2 minutes
                    time.sleep(wait_sec)
                    cached = _try_load_cached()
                    if cached is not None:
                        return cached
                    wait_sec = min(2.0, wait_sec * 1.2)
                if self._is_main_process:
                    print("Warning: index lock wait timeout, rebuilding index.")
            else:
                if self._is_main_process:
                    print("Building JSONL index (rank0)...")

        offsets = []
        offset = 0
        limit = self.max_samples if (self.max_samples is not None and not self.index_cache) else None
        with self.jsonl_path.open("rb") as f:
            for line in f:
                offsets.append(offset)
                offset += len(line)
                if limit is not None and len(offsets) >= limit:
                    break

        offsets = np.asarray(offsets, dtype=np.int64)
        if self.index_cache and limit is None:
            try:
                np.save(idx_path, offsets)
                with meta_path.open("w") as f:
                    json.dump(
                        {"file_size": file_size, "file_mtime": file_mtime, "count": int(len(offsets))},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as exc:
                if self._is_main_process:
                    print(f"Warning: failed to save index cache: {exc}")
            finally:
                try:
                    if lock_path.exists():
                        lock_path.unlink()
                except Exception:
                    pass

        return offsets

    def _get_sample_by_index(self, idx):
        if not self.lazy_load:
            return self.samples[idx]
        if self._fp is None:
            self._fp = self.jsonl_path.open("rb")
        try:
            offset = int(self._offsets[idx])
        except Exception as exc:
            raise IndexError(f"Invalid index {idx}") from exc
        self._fp.seek(offset)
        line = self._fp.readline()
        if not line:
            raise IndexError(f"Failed to read line at index {idx}")
        return json.loads(line.decode("utf-8"))

    def _build_question_text(self, question, options):
        options_text = ""
        if self.include_options and options:
            options_text = "\n".join(options)
        if self.prompt_template:
            try:
                return self.prompt_template.format(question=question, options=options_text)
            except Exception:
                # Fallback to raw question on template errors
                return f"{question}\n\n{options_text}".strip()
        if not options_text:
            return question
        return f"{question}\n\n{options_text}"

    def _normalize_region_refs(self, refs, cameras):
        if not refs:
            return []
        norm = []
        for ref in refs:
            view_idx = None
            obj_idx = None
            if isinstance(ref, dict):
                view_idx = ref.get("view_idx", ref.get("view"))
                obj_idx = ref.get("obj_idx", ref.get("obj"))
            elif isinstance(ref, (list, tuple)) and len(ref) >= 2:
                view_idx, obj_idx = ref[0], ref[1]
            if isinstance(view_idx, str) and cameras:
                if view_idx in cameras:
                    view_idx = cameras.index(view_idx)
            if view_idx is None or obj_idx is None:
                continue
            try:
                view_idx = int(view_idx)
                obj_idx = int(obj_idx)
            except Exception:
                continue
            norm.append((view_idx, obj_idx))
        return norm

    def _expand_region_tokens(self, text, region_token_counts):
        if not region_token_counts:
            return text
        expanded = text
        for count in region_token_counts:
            count = max(0, int(count))
            expanded = expanded.replace(
                self.region_placeholder,
                "[" + (self.region_token * count) + "]",
                1
            )
        return expanded

    def _infer_region_token_counts(self, region_refs, mask_nums_per_view, num_placeholders):
        if not region_refs or not mask_nums_per_view:
            return [self.region_token_num] * num_placeholders
        counts = []
        for view_idx, obj_idx in region_refs:
            count = self.region_token_num
            if 0 <= view_idx < len(mask_nums_per_view):
                view_counts = mask_nums_per_view[view_idx]
                if 0 <= obj_idx < len(view_counts):
                    try:
                        count = int(view_counts[obj_idx])
                    except Exception:
                        count = self.region_token_num
            counts.append(count)
        if len(counts) < num_placeholders:
            counts.extend([self.region_token_num] * (num_placeholders - len(counts)))
        return counts[:num_placeholders]

    def _apply_max_objects(self, sample, view_objects):
        max_keep = self.max_objects_per_sample
        if max_keep is None or max_keep <= 0:
            return view_objects, sample.get("region_refs") or []
        region_refs = sample.get("region_refs") or []
        keep_by_view = [set() for _ in view_objects]
        for ref in region_refs:
            try:
                v_idx, obj_idx = int(ref[0]), int(ref[1])
            except Exception:
                continue
            if 0 <= v_idx < len(view_objects) and 0 <= obj_idx < len(view_objects[v_idx]):
                keep_by_view[v_idx].add(obj_idx)

        new_view_objects = []
        idx_maps = []
        for v_idx, objs in enumerate(view_objects):
            if len(objs) <= max_keep:
                new_view_objects.append(objs)
                idx_maps.append({i: i for i in range(len(objs))})
                continue
            keep = list(sorted(keep_by_view[v_idx]))
            for i in range(len(objs)):
                if len(keep) >= max_keep:
                    break
                if i not in keep:
                    keep.append(i)
            new_view_objects.append([objs[i] for i in keep])
            idx_maps.append({old_i: new_i for new_i, old_i in enumerate(keep)})

        new_region_refs = []
        for ref in region_refs:
            try:
                v_idx, obj_idx = int(ref[0]), int(ref[1])
            except Exception:
                continue
            if v_idx >= len(idx_maps) or obj_idx not in idx_maps[v_idx]:
                raise ValueError("Referenced object dropped by max_objects_per_sample cap")
            new_region_refs.append([v_idx, idx_maps[v_idx][obj_idx]])

        return new_view_objects, new_region_refs

    def _load_inline_views(self, sample):
        views = sample.get("views") or []
        if not views:
            raise ValueError("Sample has no views for inline loading")

        pil_images = []
        view_ids = []
        view_objects = []

        # Prefer per-view image_path; fall back to video_path + frame_id
        frame_id_default = sample.get("frame_id", sample.get("frame_idx", 0))
        for v_idx, view in enumerate(views):
            view_id = view.get("view_id", v_idx)
            view_ids.append(view_id)

            image = None
            image_path = view.get("image_path")
            if image_path:
                try:
                    image = self._load_image(image_path)
                except FileNotFoundError:
                    image = None

            if image is None:
                if self.image_only:
                    raise FileNotFoundError(f"Image not found and image_only is set for view {v_idx}: {image_path}")
                video_path = view.get("video_path")
                frame_id = view.get("frame_id", frame_id_default)
                if video_path is None:
                    raise ValueError(f"Missing image_path/video_path for view {v_idx}")
                video_path = self._resolve_image_path(video_path)
                image = self.load_video_frame(video_path, int(frame_id))

            pil_images.append(image)
            view_objects.append(view.get("objects", []))

        view_objects, region_refs = self._apply_max_objects(sample, view_objects)

        max_objects = max((len(objs) for objs in view_objects), default=0)
        if not self.load_masks:
            return pil_images, None, view_ids, max_objects, frame_id_default, region_refs, view_objects

        masks_per_view = []
        for image, objs in zip(pil_images, view_objects):
            H, W = image.height, image.width
            masks_list = []
            for obj in objs:
                mask = decode_mask(obj.get("mask_rle"))
                if mask is None:
                    mask = _bbox_to_mask(obj.get("bbox_xyxy"), H, W)
                if mask.shape[:2] != (H, W):
                    mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                if self.split == "train" and self.mask_jitter:
                    mask = jitter_mask(
                        mask,
                        prob=self.mask_jitter_prob,
                        max_shift=self.mask_jitter_max_shift,
                        max_kernel=self.mask_jitter_max_kernel,
                    )
                masks_list.append(mask.astype(np.float32))

            # Pad missing objects to align across views (global token stays last)
            while len(masks_list) < max_objects:
                masks_list.append(np.zeros((H, W), dtype=np.float32))

            # Append global mask
            if max_objects == 0:
                masks = np.ones((1, H, W), dtype=np.float32)
            else:
                global_mask = np.ones((H, W), dtype=np.float32)
                masks = np.stack(masks_list + [global_mask], axis=0)
            masks_per_view.append(masks)

        return pil_images, masks_per_view, view_ids, max_objects, frame_id_default, region_refs, view_objects

    def __getitem__(self, idx):
        base_idx = idx
        sample = self._get_sample_by_index(idx)
        last_err = None
        for attempt in range(self.max_retries):
            sample_id = sample.get("sample_id")
            view_paths = []
            try:
                view_paths = [v.get("image_path") for v in sample.get("views", []) if v.get("image_path")]
            except Exception:
                view_paths = []
            if sample_id in self.bad_samples or any(vp in self.bad_videos for vp in view_paths):
                new_idx = self._resample_idx(base_idx, attempt)
                sample = self._get_sample_by_index(new_idx)
                continue
            try:
                result = self._load_sample(sample, idx)
                timing = result.pop("_timing", None)
                if timing and self.debug_timing and self._is_main_process:
                    worker_info = get_worker_info()
                    if (worker_info is None or worker_info.id == 0) and self._timing_logged < self.timing_max_logs:
                        if self.timing_every <= 1 or idx % self.timing_every == 0:
                            self._timing_logged += 1
                            print(
                                f"[data timing] idx={idx} "
                                f"views={timing.get('num_views')} objects={timing.get('num_objects')} "
                                f"masks={timing.get('total_masks')} "
                                f"load={timing.get('load_views'):.3f}s "
                                f"proc={timing.get('processor'):.3f}s "
                                f"crop={timing.get('additional'):.3f}s "
                                f"total={timing.get('total'):.3f}s "
                                f"sample_id={sample.get('sample_id')}"
                            )
                return result
            except Exception as e:
                last_err = e
                # Record bad sample/video to avoid repeated stalls
                if sample_id is not None:
                    self.bad_samples.add(sample_id)
                    if sample_id not in self.warned_samples:
                        if self._is_main_process:
                            print(f"Warning: Failed to load sample {idx} (attempt {attempt + 1}/{self.max_retries}): {e}")
                        self.warned_samples.add(sample_id)
                else:
                    if self._is_main_process:
                        print(f"Warning: Failed to load sample {idx} (attempt {attempt + 1}/{self.max_retries}): {e}")
                msg = str(e)
                if " from " in msg:
                    bad_path = msg.split(" from ", 1)[1].strip()
                    self.bad_videos.add(bad_path)
                    if bad_path not in self.warned_videos:
                        if self._is_main_process:
                            print(f"Warning: Mark bad video {bad_path}")
                        self.warned_videos.add(bad_path)
                new_idx = self._resample_idx(base_idx, attempt)
                sample = self._get_sample_by_index(new_idx)
        raise RuntimeError(f"Too many failures loading samples. Last error: {last_err}")

    def _resample_idx(self, base_idx, attempt):
        if self.resample_strategy in ("next", "sequential", "forward"):
            return (base_idx + 1 + attempt) % len(self)
        return random.randint(0, len(self) - 1)

    def _load_sample(self, sample, idx):
        if not sample.get("views"):
            raise ValueError("Sample missing views (combined JSONL required)")
        trace_on = False
        if self.debug_trace:
            worker_info = get_worker_info()
            if (worker_info is None or worker_info.id == 0):
                if self._trace_logged < self.debug_trace_max:
                    if self.debug_trace_every <= 1 or (idx % self.debug_trace_every == 0):
                        trace_on = True
                        self._trace_logged += 1
        if trace_on:
            print(
                f"[trace] idx={idx} sample_id={sample.get('sample_id')} start",
                flush=True
            )
        t0 = time.perf_counter()
        if trace_on:
            print(f"[trace] idx={idx} load_views start", flush=True)
        pil_images, masks_per_view, cameras, num_objects, frame_id, region_refs_raw, view_objects = self._load_inline_views(sample)
        t1 = time.perf_counter()
        if trace_on:
            print(f"[trace] idx={idx} load_views done {t1 - t0:.3f}s", flush=True)
            try:
                counts = [len(v) for v in view_objects]
            except Exception:
                counts = []
            print(f"[trace] idx={idx} objects_per_view={counts} max_keep={self.max_objects_per_sample}", flush=True)
        selected_object_names = sample.get("selected_object_names", [])
        take_name = sample.get("take_name") or sample.get("scene") or sample.get("scenario")
        task_type = str(sample.get("question_type") or "")
        task_type_upper = task_type.upper()

        # For non-Q2/Q9, only resample if a referenced object mask is empty.
        if masks_per_view is not None and task_type_upper not in ("Q2", "Q9"):
            ref_set = set()
            for ref in (region_refs_raw or []):
                try:
                    v_idx, obj_idx = int(ref[0]), int(ref[1])
                except Exception:
                    continue
                ref_set.add((v_idx, obj_idx))
            for v_idx, view_masks in enumerate(masks_per_view):
                if view_masks is None:
                    continue
                if view_masks.shape[0] <= 1:
                    continue
                obj_masks = view_masks[:-1]
                if obj_masks.size == 0:
                    continue
                sums = obj_masks.reshape(obj_masks.shape[0], -1).sum(axis=1)
                for ref_v, ref_o in ref_set:
                    if ref_v != v_idx:
                        continue
                    if ref_o < 0 or ref_o >= sums.shape[0]:
                        raise ValueError("Referenced object index out of range")
                    if sums[ref_o] <= 0:
                        raise ValueError("Referenced object mask empty for non-Q2/Q9 sample")

        processed = None
        if self.use_processor:
            dummy_text = [""] * len(pil_images)
            if trace_on:
                print(f"[trace] idx={idx} processor start", flush=True)
            processed = self.processor(text=dummy_text, images=pil_images, return_tensors="pt")
            t2 = time.perf_counter()
            if trace_on:
                print(f"[trace] idx={idx} processor done {t2 - t1:.3f}s", flush=True)
            num_views = processed["image_grid_thw"].shape[0]
            if num_views < 2 or len(pil_images) < 2:
                raise ValueError(
                    f"Invalid number of views: images={len(pil_images)}, pixel_values={processed['pixel_values'].shape}, "
                    f"sample_id={sample.get('sample_id')}, take={sample.get('take_name')}, frame={frame_id}"
                )
            if num_views != len(pil_images):
                raise ValueError(
                    f"Processor dropped views: images={len(pil_images)}, pixel_values={processed['pixel_values'].shape}, "
                    f"sample_id={sample.get('sample_id')}, take={sample.get('take_name')}, frame={frame_id}"
                )
        else:
            t2 = time.perf_counter()
            num_views = len(pil_images)
            if num_views < 2:
                raise ValueError(
                    f"Invalid number of views: images={len(pil_images)}, "
                    f"sample_id={sample.get('sample_id')}, take={sample.get('take_name')}, frame={frame_id}"
                )

        question_text = self._build_question_text(sample.get("question", ""), sample.get("options", []))
        region_refs = self._normalize_region_refs(region_refs_raw, cameras)
        region_token_counts = sample.get("region_token_counts")

        if masks_per_view is not None and len(masks_per_view) != num_views:
            raise ValueError(
                f"Masks/view mismatch: masks={len(masks_per_view)}, views={num_views}, "
                f"sample_id={sample.get('sample_id')}, take={sample.get('take_name')}, frame={frame_id}"
            )

        if self.use_additional_inputs:
            if trace_on:
                print(f"[trace] idx={idx} additional start", flush=True)
            additional_pixel_values, additional_grid_thw, additional_masks, additional_box_params, additional_mask_nums = prepare_additional_inputs(
                pil_images,
                masks_per_view,
                self.processor,
                patch_size=14,
                max_tokens=self.mask_num,
                return_mask_nums=True,
            )
            t3 = time.perf_counter()
            if trace_on:
                print(f"[trace] idx={idx} additional done {t3 - t2:.3f}s", flush=True)
        else:
            additional_pixel_values = None
            additional_grid_thw = None
            additional_masks = None
            additional_box_params = None
            additional_mask_nums = None
            t3 = t2

        num_placeholders = question_text.count(self.region_placeholder)
        if num_placeholders > 0 and not self.defer_region_token_expansion:
            if region_token_counts is None:
                region_token_counts = self._infer_region_token_counts(
                    region_refs,
                    additional_mask_nums,
                    num_placeholders
                )
            if len(region_token_counts) != num_placeholders:
                if len(region_token_counts) < num_placeholders:
                    region_token_counts = region_token_counts + [self.region_token_num] * (num_placeholders - len(region_token_counts))
                else:
                    region_token_counts = region_token_counts[:num_placeholders]
            question_text = self._expand_region_tokens(question_text, region_token_counts)

        compute_infonce = bool(sample.get("compute_infonce", False))
        if num_objects == 0 or task_type_upper in ("Q2", "Q9"):
            compute_infonce = False

        metadata = sample.get("metadata", {})
        spatial_info = sample.get("spatial_info", {})
        target_index = None
        if task_type == "reid":
            object_name = metadata.get("object_name")
            if not object_name:
                object_name = infer_target_object_name(metadata, spatial_info)
            if object_name and object_name in selected_object_names:
                target_index = selected_object_names.index(object_name)
            else:
                target_index = 0 if num_objects > 0 else 0
        else:
            target_index = num_objects  # global token index (K objects + 1 global)

        object_ids = sample.get("object_ids")
        if object_ids is None and selected_object_names:
            if take_name:
                object_ids = [name if ":" in name else f"{take_name}:{name}" for name in selected_object_names]
            else:
                object_ids = list(selected_object_names)
        if object_ids is None and view_objects:
            # Fallback: build object ids from dataset + scene + track_id so
            # sample-local track ids do not collide across different scenes.
            try:
                objs_a = view_objects[0] if view_objects else []
                dataset_name = sample.get("dataset") or "unknown_dataset"
                scene_name = sample.get("scene") or take_name or "unknown_scene"
                object_ids = [
                    f"{dataset_name}:{scene_name}:{o.get('track_id', i)}"
                    for i, o in enumerate(objs_a)
                ]
            except Exception:
                object_ids = None

        meta = dict(sample.get("metadata", {}))
        meta.update({
            "sample_id": sample.get("sample_id"),
            "take_name": take_name,
            "frame_id": frame_id,
            "question_type": sample.get("question_type"),
            "num_objects": num_objects,
            "cameras": cameras,
            "dataset": sample.get("dataset"),
            "scene": sample.get("scene"),
        })

        result = {
            "pixel_values": processed["pixel_values"] if processed is not None else None,
            "image_grid_thw": processed["image_grid_thw"] if processed is not None else None,
            "masks": masks_per_view,
            "additional_pixel_values": additional_pixel_values,
            "additional_grid_thw": additional_grid_thw,
            "additional_masks": additional_masks,
            "additional_box_params": additional_box_params,
            "additional_mask_nums": additional_mask_nums,
            "question": question_text,
            "answer": sample.get("answer"),
            "compute_infonce": compute_infonce,
            "target_index": target_index,
            "object_ids": object_ids,
            "region_refs": region_refs,
            "region_token_counts": region_token_counts,
            "metadata": meta,
            "_timing": {
                "load_views": t1 - t0,
                "processor": t2 - t1,
                "additional": t3 - t2,
                "total": t3 - t0,
                "num_views": int(num_views),
                "num_objects": int(num_objects),
                "total_masks": int(sum(len(v) for v in masks_per_view)) if masks_per_view is not None else 0,
            },
        }
        if self.return_raw_images:
            result["raw_images"] = pil_images
        return result


def collate_fn(batch):
    """
    Collate function for JSONL dataset.
    """
    pixel_values = [item["pixel_values"] for item in batch]
    image_grid_thw = [item["image_grid_thw"] for item in batch]
    masks = [item["masks"] for item in batch] if batch[0]["masks"] is not None else None
    timings = [item.get("_timing") for item in batch]
    additional_pixel_values = [item.get("additional_pixel_values") for item in batch]
    additional_grid_thw = [item.get("additional_grid_thw") for item in batch]
    additional_masks = [item.get("additional_masks") for item in batch]
    additional_box_params = [item.get("additional_box_params") for item in batch]
    additional_mask_nums = [item.get("additional_mask_nums") for item in batch]
    raw_images = [item.get("raw_images") for item in batch]
    metadata = [item.get("metadata", {}) for item in batch]

    result = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "masks": masks,
        "additional_pixel_values": additional_pixel_values,
        "additional_grid_thw": additional_grid_thw,
        "additional_masks": additional_masks,
        "additional_box_params": additional_box_params,
        "additional_mask_nums": additional_mask_nums,
        "raw_images": raw_images,
        "metadata": metadata,
        "questions": [item["question"] for item in batch],
        "answers": [item.get("answer") for item in batch],
        "compute_infonce": [item.get("compute_infonce", False) for item in batch],
        "target_indices": [item.get("target_index", None) for item in batch],
        "object_ids": [item.get("object_ids", None) for item in batch],
        "region_refs": [item.get("region_refs") for item in batch],
        "region_token_counts": [item.get("region_token_counts") for item in batch],
        "timing": timings,
    }

    return result
