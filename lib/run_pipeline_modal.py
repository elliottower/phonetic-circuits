"""Modal wrapper for run_pipeline.py.

Usage:
    modal run --detach lib/run_pipeline_modal.py
    modal run --detach lib/run_pipeline_modal.py --tasks op4_oronym
    modal run --detach lib/run_pipeline_modal.py --model gpt2-medium
"""
import modal

app = modal.App("phonetic-circuits-eap-ig")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformer-lens",
        "transformers",
        "einops",
        "pandas",
        "numpy<2",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "jaxtyping",
        "typeguard",
    )
    .add_local_dir("datasets", "/root/phonetic-circuits/datasets")
    .add_local_dir("lib", "/root/phonetic-circuits/lib")
    .add_local_dir("reference/mib/EAP-IG/src/eap", "/root/phonetic-circuits/reference/mib/EAP-IG/src/eap")
    .add_local_dir("reference/mib/MIB_circuit_track", "/root/phonetic-circuits/reference/mib/MIB_circuit_track")
)

vol = modal.Volume.from_name("phonetic-circuits-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/results": vol},
)
def run_eap_ig(tasks: list[str] | None = None, model: str = "gpt2"):
    import subprocess
    import shutil

    cmd = [
        "python", "-m", "lib.run_pipeline",
        "--model", model,
        "--output-dir", "/results",
        "--device", "cuda",
        "--batch-size", "64",
    ]
    if tasks:
        cmd += ["--tasks"] + tasks

    result = subprocess.run(
        cmd,
        cwd="/root/phonetic-circuits",
        capture_output=False,
    )
    vol.commit()
    return result.returncode


@app.local_entrypoint()
def main(
    tasks: str = "",
    model: str = "gpt2",
):
    task_list = [t.strip() for t in tasks.split(",") if t.strip()] if tasks else None
    rc = run_eap_ig.remote(tasks=task_list, model=model)
    print(f"Exit code: {rc}")
