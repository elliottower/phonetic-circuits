"""Launch ALL phonetic circuit experiments in parallel on Modal.

Each experiment gets its own A10G container. All results go to the
shared phonetic-circuits-results volume.

Usage:
    modal run --detach lib/run_all_experiments_modal.py
    modal run --detach lib/run_all_experiments_modal.py --skip causal_discovery
"""
import modal

app = modal.App("phonetic-all-experiments")
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
    .add_local_dir("results", "/root/phonetic-circuits/results")
)

TASKS = ["op1_hypocorism", "op2_clipping", "op3_initialism",
         "op4_oronym", "op5_homophone", "op6_folk_etym"]


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/results": vol})
def run_das_causal_graph():
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [sys.executable, "-m", "lib.run_das_causal_graph",
           "--retrain", "--num-examples", "200", "--n-steps", "100",
           "--output-dir", "/results/das_causal_graph", "--device", "cuda"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return "das_causal_graph: done"


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/results": vol})
def run_compositional_ablation(circuit_a: str, circuit_b: str):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [sys.executable, "-m", "lib.run_compositional_ablation",
           "--circuit-a", circuit_a, "--circuit-b", circuit_b,
           "--top-k", "15", "--num-examples", "100",
           "--output-dir", "/results/compositional_ablation", "--device", "cuda"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return f"compositional_ablation ({circuit_a} x {circuit_b}): done"


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/results": vol})
def run_das_transfer(layer: int = 10):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [sys.executable, "-m", "lib.run_das_transfer",
           "--layer", str(layer), "--k", "1",
           "--output-dir", "/results/das_transfer", "--device", "cuda"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return f"das_transfer (L{layer}): done"


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/results": vol})
def run_causal_discovery():
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [sys.executable, "-m", "lib.run_causal_discovery",
           "--top-k", "30",
           "--output-dir", "/results/causal_discovery", "--device", "cuda"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return "causal_discovery: done"


@app.function(image=image, gpu="A10G", timeout=14400, volumes={"/results": vol})
def run_das_composition(layer: int = 10):
    import subprocess
    import sys
    import os

    os.chdir("/root/phonetic-circuits")
    sys.path.insert(0, "/root/phonetic-circuits")

    cmd = [sys.executable, "-m", "lib.run_das_composition",
           "--retrain", "--layer", str(layer),
           "--output-dir", "/results/das_composition", "--device", "cuda"]
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    return f"das_composition (L{layer}): done"


@app.local_entrypoint()
def main(skip: str = ""):
    skip_set = {s.strip() for s in skip.split(",") if s.strip()}

    futures = []

    if "das_causal_graph" not in skip_set:
        print("Launching: DAS causal graph discovery (LiNGAM on DAS scalars)")
        futures.append(run_das_causal_graph.spawn())

    if "compositional_ablation" not in skip_set:
        pairs = [
            ("op1_hypocorism", "op4_oronym"),
            ("op1_hypocorism", "op2_clipping"),
            ("op4_oronym", "op5_homophone"),
        ]
        for a, b in pairs:
            print(f"Launching: Compositional ablation ({a} x {b})")
            futures.append(run_compositional_ablation.spawn(a, b))

    if "das_transfer" not in skip_set:
        for layer in [9, 10, 11]:
            print(f"Launching: DAS transfer (layer {layer})")
            futures.append(run_das_transfer.spawn(layer))

    if "causal_discovery" not in skip_set:
        print("Launching: Causal discovery (PC/CD-NOD/TPC/LiNGAM on head activations)")
        futures.append(run_causal_discovery.spawn())

    if "das_composition" not in skip_set:
        for layer in [9, 10]:
            print(f"Launching: DAS composition (layer {layer})")
            futures.append(run_das_composition.spawn(layer))

    print(f"\n{len(futures)} containers launched. Waiting for results...")
    for f in futures:
        result = f.get()
        print(f"  {result}")

    print("\nAll experiments complete. Results in Modal volume: phonetic-circuits-results")
