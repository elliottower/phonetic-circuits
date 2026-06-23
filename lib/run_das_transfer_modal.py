"""Thin Modal wrapper for cross-task DAS transfer.

Usage:
    modal run --detach lib/run_das_transfer_modal.py
    modal run --detach lib/run_das_transfer_modal.py --layer 8 --k 2
"""
import modal

app = modal.App("phonetic-das-transfer")
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


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/results": vol})
def run(layer: int = 10, k: int = 1):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [
        sys.executable, "-m", "lib.run_das_transfer",
        "--layer", str(layer),
        "--k", str(k),
        "--output-dir", "/results/das_transfer",
        "--device", "cuda",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    print(f"Done, results committed to volume.", flush=True)


@app.local_entrypoint()
def main(layer: int = 10, k: int = 1):
    run.remote(layer=layer, k=k)
