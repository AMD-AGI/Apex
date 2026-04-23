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


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-name", default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--timed-runs", type=int, default=2)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-dir", default="/tmp/fastvideo_e2e_outputs")
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

    timed_runs = []
    for i in range(args.timed_runs):
        t0 = time.perf_counter()
        _ = generator.generate_video(
            PROMPTS[(i + 1) % len(PROMPTS)],
            output_path=str(output_dir),
            save_video=False,
            sampling_param=sampling_param,
        )
        _sync()
        t1 = time.perf_counter()
        timed_runs.append(t1 - t0)

    result = {
        "repo_root": str(repo_root),
        "model_name": args.model_name,
        "attention_backend": os.environ.get("FASTVIDEO_ATTENTION_BACKEND", ""),
        "load_time_s": load_t1 - load_t0,
        "warmup_time_s": warm_t1 - warm_t0,
        "timed_runs_s": timed_runs,
        "timed_mean_s": sum(timed_runs) / len(timed_runs) if timed_runs else 0.0,
        "timed_min_s": min(timed_runs) if timed_runs else 0.0,
    }

    payload = json.dumps(result, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
