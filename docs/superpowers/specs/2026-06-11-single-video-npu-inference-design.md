# Single-Video NPU Inference Design

## Goal

Add a standalone command-line flow that loads a VideoMAE V2 classification
checkpoint, runs one video on a single Ascend NPU, and prints Top-K class
probabilities.

## Scope

The implementation will add `run_single_video_inference.py`. It will not
modify the training or dataset evaluation flows, add distributed inference,
or introduce a serving framework.

## Command-Line Interface

Required arguments:

- `--video`: input video path.
- `--checkpoint`: classification checkpoint path.
- `--labels`: UTF-8 text file containing one class name per line. The line
  number is the zero-based class ID.
- `--model`: a model name registered by the existing `models` package.
- `--num-classes`: classification head size.

Optional arguments:

- `--device`, default `npu:0`; `cpu` is supported for functional debugging.
- `--num-frames`, default `16`.
- `--sampling-rate`, default `4`.
- `--input-size`, default `224`.
- `--short-side-size`, default `224`.
- `--tubelet-size`, default `2`.
- `--top-k`, default `5`.
- Model construction options already needed by existing VideoMAE checkpoints,
  such as `--drop-path` and `--use-mean-pooling`.

The script will reject missing files, non-positive numeric values, a label
count different from `--num-classes`, and a `--top-k` larger than the number
of classes.

## Components

### Video Decoding

The decoder will try Decord first. If Decord cannot be imported or cannot
decode the video, it will retry with OpenCV and report the selected backend.
Both backends return RGB frames as a NumPy array.

Decoding remains on CPU. Ascend NPU video decoding is outside this task.

### Temporal Sampling

Inference uses one deterministic center clip:

1. Compute the desired span from `num_frames * sampling_rate`.
2. Center that span within the available video.
3. Select `num_frames` indices at the configured sampling rate.
4. Clamp indices for videos shorter than the requested span, repeating edge
   frames when necessary.

This provides stable results and avoids the cost of multi-segment evaluation.

### Spatial Preprocessing

Frames will be:

1. Converted to RGB by the decoder adapter.
2. Resized while preserving aspect ratio so the short side equals
   `short_side_size`.
3. Center-cropped to `input_size`.
4. Converted to float in `[0, 1]`.
5. Normalized with the ImageNet mean and standard deviation used by the
   existing fine-tuning pipeline.
6. Rearranged to `[1, C, T, H, W]`.

### Model and Checkpoint Loading

The script imports `models` to register VideoMAE model names and constructs the
model through `timm.models.create_model`.

Checkpoint loading supports:

- A bare state dictionary.
- A dictionary containing `model`, `module`, or `state_dict`.
- Parameter names prefixed with `module.` or `backbone.`.

The loader uses strict shape validation. A classification head whose output
dimension differs from `--num-classes` is an error instead of being silently
discarded.

### Device Execution

For an NPU device, the script imports `torch_npu`, validates NPU availability,
sets the selected device, moves the model and input tensor to it, and runs
under `torch.inference_mode()`.

The implementation will not enable CUDA AMP or DeepSpeed. Initial inference
uses float32 for predictable compatibility. CPU mode follows the same path
without importing `torch_npu`.

### Output

The logits are converted with softmax. Results are printed in descending
order with rank, zero-based class ID, class name, and probability.

Example:

```text
decoder: opencv
device: npu:0
1  123  playing guitar  0.812345
2  117  strumming guitar 0.083210
```

## Error Handling

Errors will identify the failing stage and exit non-zero:

- Invalid input paths or label count.
- Both video decoders unavailable or unable to decode.
- Unknown model name.
- Unsupported checkpoint structure or incompatible parameter shapes.
- Missing `torch_npu`, unavailable NPU, or invalid NPU device.
- Model output shape inconsistent with `--num-classes`.

Decoder fallback is the only recoverable failure and will be reported rather
than hidden.

## Testing

CPU unit tests will cover:

- Center-clip indices for long, exact-length, and short videos.
- Label loading and validation.
- Checkpoint state-dictionary extraction and prefix removal.
- Spatial preprocessing output shape and deterministic center crop.
- Top-K result ordering and label association.
- Decoder fallback using controlled decoder adapters.

The NPU integration check will run the CLI with a real checkpoint and video,
verify successful model execution on `npu:0`, and confirm Top-K output. This
check requires the target Ascend host and model assets and cannot be replaced
by a CPU unit test.

## Completion Criteria

- A single command runs one video through a supported classification
  checkpoint on `npu:0`.
- Output includes class IDs, names, and probabilities in descending order.
- Decord is preferred and OpenCV fallback works.
- CPU unit tests pass.
- The CLI provides clear errors for invalid labels, checkpoints, videos, and
  unavailable NPU devices.
