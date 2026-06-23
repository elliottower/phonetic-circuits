"""Thin Modal wrapper for causal discovery on head activations.

Usage:
    modal run --detach lib/run_causal_discovery_modal.py
    modal run --detach lib/run_causal_discovery_modal.py --top-k 20
"""
import modal

app = modal.App("phonetic-causal-discovery")
vol = modal.Volume.from_name("phonetic-circuits-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformer-lens", "transformers", "einops", "pandas",
        "numpy<2", "scipy", "scikit-learn", "matplotlib", "tqdm",
        "jaxtyping", "typeguard",
        "causal-learn", "lingam", "tigramite",
    )
    .add_local_dir("datasets", "/root/phonetic-circuits/datasets")
    .add_local_dir("lib", "/root/phonetic-circuits/lib")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/results": vol},
)
def run(top_k: int = 30):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [
        sys.executable, "-m", "lib.run_causal_discovery",
        "--top-k", str(top_k),
        "--output-dir", "/results/causal_discovery",
        "--device", "cuda",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    print("Done, results committed to volume.", flush=True)


@app.local_entrypoint()
def main(top_k: int = 30):
    run.remote(top_k=top_k)
