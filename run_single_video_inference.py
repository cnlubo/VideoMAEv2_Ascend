#!/usr/bin/env python3
"""Run VideoMAE V2 classification on one video with one Ascend NPU."""

import argparse
import math
from collections import OrderedDict
from pathlib import Path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_KEYS = ("model", "module", "state_dict")
STATE_DICT_PREFIXES = ("_orig_mod.", "module.", "backbone.", "encoder.")


def get_args():
    parser = argparse.ArgumentParser(
        description="VideoMAE V2 single-video classification")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-frames", default=16, type=int)
    parser.add_argument("--sampling-rate", default=4, type=int)
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--short-side-size", default=224, type=int)
    parser.add_argument("--tubelet-size", default=2, type=int)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--drop-path", default=0.0, type=float)
    parser.add_argument("--init-scale", default=0.001, type=float)
    parser.add_argument(
        "--use-mean-pooling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args):
    for name in ("video", "checkpoint", "labels"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")

    positive_values = {
        "num_classes": args.num_classes,
        "num_frames": args.num_frames,
        "sampling_rate": args.sampling_rate,
        "input_size": args.input_size,
        "short_side_size": args.short_side_size,
        "tubelet_size": args.tubelet_size,
        "top_k": args.top_k,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero, got {value}")

    if args.top_k > args.num_classes:
        raise ValueError(
            f"top_k ({args.top_k}) cannot exceed num_classes "
            f"({args.num_classes})")
    if args.num_frames % args.tubelet_size != 0:
        raise ValueError(
            "num_frames must be divisible by tubelet_size: "
            f"{args.num_frames} vs {args.tubelet_size}")
    if args.short_side_size < args.input_size:
        raise ValueError(
            "short_side_size must be greater than or equal to input_size: "
            f"{args.short_side_size} vs {args.input_size}")


def load_labels(path, num_classes):
    labels = path.read_text(encoding="utf-8").splitlines()
    for line_number, label in enumerate(labels, start=1):
        if not label.strip():
            raise ValueError(f"empty class name at line {line_number}: {path}")
    if len(labels) != num_classes:
        raise ValueError(
            f"label count ({len(labels)}) does not match num_classes "
            f"({num_classes}): {path}")
    return labels


def get_center_clip_indices(video_length, num_frames, sampling_rate):
    if video_length <= 0:
        raise ValueError(f"video has no decodable frames: {video_length}")

    clip_span = (num_frames - 1) * sampling_rate + 1
    start = (video_length - clip_span) // 2
    indices = [start + index * sampling_rate for index in range(num_frames)]
    return [min(max(index, 0), video_length - 1) for index in indices]


def _decord_video_length(path):
    from decord import VideoReader, cpu

    return len(VideoReader(str(path), ctx=cpu(0), num_threads=1))


def _decode_with_decord(path, indices):
    from decord import VideoReader, cpu

    video_reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return video_reader.get_batch(indices).asnumpy()


def _opencv_video_length(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError("cannot open video")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"invalid frame count: {frame_count}")
        return frame_count
    finally:
        capture.release()


def _decode_with_opencv(path, indices):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("cannot open video")

    frames_by_index = {}
    try:
        for frame_index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"cannot decode frame {frame_index}")
            frames_by_index[frame_index] = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    return np.stack([frames_by_index[index] for index in indices], axis=0)


def decode_video(path, num_frames, sampling_rate):
    backends = (
        ("decord", _decord_video_length, _decode_with_decord),
        ("opencv", _opencv_video_length, _decode_with_opencv),
    )
    errors = []
    for backend, length_reader, decoder in backends:
        try:
            video_length = length_reader(path)
            indices = get_center_clip_indices(
                video_length=video_length,
                num_frames=num_frames,
                sampling_rate=sampling_rate,
            )
            return decoder(path, indices), backend
        except Exception as error:
            errors.append(f"{backend}: {error}")

    raise RuntimeError(
        f"failed to decode video {path}; " + "; ".join(errors))


def preprocess_frames(frames, input_size, short_side_size):
    import torch
    import torch.nn.functional as functional

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "decoded frames must have shape [T, H, W, 3], "
            f"got {tuple(frames.shape)}")
    if frames.shape[0] == 0:
        raise ValueError("decoded video contains no frames")

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    height, width = tensor.shape[-2:]
    scale = short_side_size / min(height, width)
    resized_height = max(short_side_size, int(round(height * scale)))
    resized_width = max(short_side_size, int(round(width * scale)))
    tensor = functional.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    top = (resized_height - input_size) // 2
    left = (resized_width - input_size) // 2
    tensor = tensor[:, :, top:top + input_size, left:left + input_size]
    if tensor.shape[-2:] != (input_size, input_size):
        raise ValueError(
            f"center crop produced invalid shape: {tuple(tensor.shape)}")

    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def _strip_state_dict_prefixes(key):
    changed = True
    while changed:
        changed = False
        for prefix in STATE_DICT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("checkpoint does not contain a state dictionary")

    state_dict = None
    for key in CHECKPOINT_KEYS:
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            state_dict = candidate
            break

    if state_dict is None:
        metadata_keys = {
            "epoch", "optimizer", "scaler", "args", "model_ema",
            "lr_scheduler"
        }
        if any(key in checkpoint for key in metadata_keys):
            raise ValueError("checkpoint does not contain a model state dictionary")
        state_dict = checkpoint

    normalized = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise ValueError("checkpoint state dictionary has a non-string key")
        normalized[_strip_state_dict_prefixes(key)] = value
    return normalized


def build_model(args):
    from timm.models import create_model

    import models  # noqa: F401

    try:
        return create_model(
            args.model,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            drop_rate=0.0,
            drop_path_rate=args.drop_path,
            attn_drop_rate=0.0,
            head_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            with_cp=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"failed to create model '{args.model}': {error}") from error


def load_checkpoint(model, checkpoint_path):
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
    except Exception as error:
        raise RuntimeError(
            f"failed to load checkpoint {checkpoint_path}: {error}") from error


def prepare_device(device_name):
    import torch

    if device_name == "cpu":
        return torch.device("cpu")
    if not device_name.startswith("npu:"):
        raise ValueError(
            f"unsupported device '{device_name}'; use 'npu:N' or 'cpu'")

    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "torch_npu is required for NPU inference") from error

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available")

    try:
        device = torch.device(device_name)
        torch.npu.set_device(device)
        return device
    except Exception as error:
        raise RuntimeError(
            f"failed to select NPU device '{device_name}': {error}") from error


def get_topk_predictions(logits, labels, top_k):
    if len(logits) != len(labels):
        raise ValueError(
            f"logit count ({len(logits)}) does not match label count "
            f"({len(labels)})")

    max_logit = max(logits)
    exponentials = [math.exp(value - max_logit) for value in logits]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    class_ids = sorted(
        range(len(probabilities)),
        key=probabilities.__getitem__,
        reverse=True,
    )[:top_k]
    return [
        {
            "class_id": class_id,
            "class_name": labels[class_id],
            "probability": probabilities[class_id],
        } for class_id in class_ids
    ]


def run_inference(args):
    import torch

    validate_args(args)
    labels = load_labels(args.labels, args.num_classes)
    frames, decoder_backend = decode_video(
        args.video,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
    )
    inputs = preprocess_frames(
        frames,
        input_size=args.input_size,
        short_side_size=args.short_side_size,
    )

    device = prepare_device(args.device)
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    model.to(device)
    inputs = inputs.to(device, non_blocking=device.type == "npu")

    with torch.inference_mode():
        outputs = model(inputs)

    if outputs.ndim != 2 or outputs.shape != (1, args.num_classes):
        raise RuntimeError(
            "model output must have shape "
            f"[1, {args.num_classes}], got {tuple(outputs.shape)}")

    logits = outputs[0].detach().float().cpu().tolist()
    predictions = get_topk_predictions(logits, labels, args.top_k)

    print(f"decoder: {decoder_backend}")
    print(f"device: {device}")
    for rank, prediction in enumerate(predictions, start=1):
        print(
            f"{rank}\t{prediction['class_id']}\t"
            f"{prediction['class_name']}\t"
            f"{prediction['probability']:.6f}")


def main():
    try:
        run_inference(get_args())
    except Exception as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
