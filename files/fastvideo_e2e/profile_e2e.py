import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


PROMPTS = [
    (
        "A neon-lit alley in futuristic Tokyo during a heavy rainstorm at night. "
        "The puddles reflect glowing signs in kanji, advertising ramen, karaoke, "
        "and VR arcades. A woman in a translucent raincoat walks briskly with an "
        "LED umbrella. Steam rises from a street food cart, and a cat darts "
        "across the screen. Raindrops are visible on the camera lens, creating "
        "a cinematic bokeh effect."
    ),
    (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently "
        "in the breeze, enhancing the lion's commanding presence. The tone is "
        "vibrant, embodying the raw energy of the wild. Low angle, steady "
        "tracking shot, cinematic."
    ),
]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-name", default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--output-dir", default="/tmp/fastvideo_profile_outputs")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    kernel_py = repo_root / "fastvideo-kernel" / "python"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(kernel_py))

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")

    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_t0 = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        args.model_name,
        num_gpus=1,
        use_fsdp_inference=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=0.8,
    )
    sampling_param = SamplingParam.from_pretrained(args.model_name)
    _sync()
    load_t1 = time.perf_counter()

    warm_t0 = time.perf_counter()
    _ = generator.generate_video(
        PROMPTS[0],
        output_path=str(output_dir),
        save_video=False,
        sampling_param=sampling_param,
    )
    _sync()
    warm_t1 = time.perf_counter()

    trace_dir = output_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir), use_gzip=True),
    ) as prof:
        profile_t0 = time.perf_counter()
        _ = generator.generate_video(
            PROMPTS[1],
            output_path=str(output_dir),
            save_video=False,
            sampling_param=sampling_param,
        )
        _sync()
        profile_t1 = time.perf_counter()
        prof.step()

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    top_events = []
    for evt in prof.key_averages():
        top_events.append(
            {
                "key": evt.key,
                "self_cpu_time_total_us": evt.self_cpu_time_total,
                "cpu_time_total_us": evt.cpu_time_total,
                "self_cuda_time_total_us": getattr(evt, "self_cuda_time_total", 0.0),
                "cuda_time_total_us": getattr(evt, "cuda_time_total", 0.0),
                "count": evt.count,
            }
        )
    top_events.sort(key=lambda x: x["self_cuda_time_total_us"], reverse=True)

    result = {
        "repo_root": str(repo_root),
        "model_name": args.model_name,
        "attention_backend": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "load_time_s": load_t1 - load_t0,
        "warmup_time_s": warm_t1 - warm_t0,
        "profiled_generate_time_s": profile_t1 - profile_t0,
        "trace_dir": str(trace_dir),
        "top_events": top_events[:80],
        "top_table": prof.key_averages().table(sort_by=sort_key, row_limit=50),
    }

    result_path = output_dir / "profile_summary.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
