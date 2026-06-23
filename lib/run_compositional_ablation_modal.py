"""Thin Modal wrapper for compositional ablation.

Usage:
    modal run --detach lib/run_compositional_ablation_modal.py
    modal run --detach lib/run_compositional_ablation_modal.py --circuit-a op1_hypocorism --circuit-b op4_oronym
"""
import modal

app = modal.App("phonetic-compositional-ablation")
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
    .add_local_dir("results", "/root/phonetic-circuits/results")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/results": vol},
)
def run(circuit_a: str = "op1_hypocorism", circuit_b: str = "op4_oronym", top_k: int = 15):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [
        sys.executable, "-m", "lib.run_compositional_ablation",
        "--circuit-a", circuit_a,
        "--circuit-b", circuit_b,
        "--top-k", str(top_k),
        "--output-dir", "/results/compositional_ablation",
        "--device", "cuda",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    print("Done, results committed to volume.", flush=True)


@app.local_entrypoint()
def main(circuit_a: str = "op1_hypocorism", circuit_b: str = "op4_oronym", top_k: int = 15):
    run.remote(circuit_a=circuit_a, circuit_b=circuit_b, top_k=top_k)
