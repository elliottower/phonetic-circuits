"""Thin Modal wrapper for DAS causal graph discovery.

Trains DAS directions per layer per task, then runs LiNGAM / PC on the
scalar causal variables to discover the DAG between layers.

Usage:
    modal run --detach lib/run_das_causal_graph_modal.py
"""
import modal

app = modal.App("phonetic-das-causal-graph")
vol = modal.Volume.from_name("phonetic-circuits-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformer-lens", "transformers", "einops", "pandas",
        "numpy<2", "scipy", "scikit-learn", "matplotlib", "tqdm",
        "jaxtyping", "typeguard",
        "causal-learn", "lingam",
    )
    .add_local_dir("datasets", "/root/phonetic-circuits/datasets")
    .add_local_dir("lib", "/root/phonetic-circuits/lib")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,
    volumes={"/results": vol},
)
def run(num_examples: int = 200, n_steps: int = 100):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [
        sys.executable, "-m", "lib.run_das_causal_graph",
        "--retrain",
        "--num-examples", str(num_examples),
        "--n-steps", str(n_steps),
        "--output-dir", "/results/das_causal_graph",
        "--device", "cuda",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    print("Done, results committed to volume.", flush=True)


@app.local_entrypoint()
def main(num_examples: int = 200, n_steps: int = 100):
    run.remote(num_examples=num_examples, n_steps=n_steps)
