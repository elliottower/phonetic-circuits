"""Thin Modal wrapper for DAS causal variable localization.

Parallelizes across tasks — one GPU container per task.

Usage:
    modal run --detach lib/run_das_modal.py
    modal run --detach lib/run_das_modal.py --k 4
"""
import modal

app = modal.App("phonetic-das")
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

TASKS = ["op1_hypocorism", "op2_clipping", "op3_initialism",
         "op4_oronym", "op5_homophone", "op7_folk_etym"]


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={"/results": vol},
)
def run_task(task: str, k: int = 1):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [
        sys.executable, "-m", "lib.run_das",
        "--tasks", task,
        "--k", str(k),
        "--output-dir", "/results/das",
        "--device", "cuda",
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    print(f"Done {task}, results committed to volume.", flush=True)


@app.local_entrypoint()
def main(k: int = 1):
    list(run_task.map(TASKS, kwargs={"k": k}))
