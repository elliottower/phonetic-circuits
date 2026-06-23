"""Thin Modal wrapper for activation patching.

Usage:
    modal run --detach lib/run_act_patching_modal.py
    modal run --detach lib/run_act_patching_modal.py --level edge
"""
import modal

app = modal.App("phonetic-act-patching")
vol = modal.Volume.from_name("phonetic-circuits-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformer-lens", "transformers", "einops", "pandas",
        "numpy<2", "scipy", "scikit-learn", "matplotlib", "tqdm",
        "jaxtyping", "typeguard",
    )
    .add_local_dir("datasets", "/root/phonetic-circuits/datasets")
    .add_local_dir("lib", "/root/phonetic-circuits/lib")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=21600,
    volumes={"/results": vol},
)
def run(level: str = "node"):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [
        sys.executable, "-m", "lib.run_act_patching",
        "--level", level,
        "--output-dir", "/results/act_patching",
        "--device", "cuda",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    print("Done, results committed to volume.", flush=True)


@app.local_entrypoint()
def main(level: str = "node"):
    run.remote(level=level)
