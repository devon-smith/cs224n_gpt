"""
Run training on Modal GPU cloud.
Usage:
  modal run modal_run.py          # submission runs only
  modal run modal_run.py --all    # submission + analysis + benchmark
"""

import modal
import subprocess

app = modal.App("cs224n-gpt")

# Docker image with all dependencies + local code copied in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "einops", "tqdm", "sacrebleu", "tokenizers", "scikit-learn", "requests", "importlib_metadata")  # v2
    .add_local_dir(
        "/Users/lilybailey/Documents/GitHub/cs224n_gpt",
        remote_path="/cs224n_gpt",
        ignore=["__pycache__", "*.pt", "*.zip", "*.pyc"],
    )
)

# Volume to persist outputs across runs
volume = modal.Volume.from_name("cs224n-outputs", create_if_missing=True)


def run(cmd: str):
    result = subprocess.run(cmd, shell=True, cwd="/cs224n_gpt")
    if result.returncode != 0:
        print(f"Command failed: {cmd}")


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60 * 12,  # 12 hours max
)
def train_submission():
    import shutil, os
    os.makedirs("/cs224n_gpt/predictions", exist_ok=True)
    os.makedirs("/cs224n_gpt/logs", exist_ok=True)

    print("=== [1/4] Classifier: last-linear-layer ===")
    run("python -u classifier.py --use_gpu --fine-tune-mode last-linear-layer --epochs 10 --lr 1e-3")

    print("=== [2/4] Classifier: full-model ===")
    run("python -u classifier.py --use_gpu --fine-tune-mode full-model --epochs 5 --lr 1e-5")

    print("=== [3/4] Paraphrase detection ===")
    run("python -u paraphrase_detection.py --use_gpu --epochs 5 --lr 1e-5")

    print("=== [4/4] Sonnet generation ===")
    run("python -u sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5")

    # Copy predictions to volume
    shutil.copytree("/cs224n_gpt/predictions", "/outputs/predictions", dirs_exist_ok=True)
    volume.commit()
    print("ALL SUBMISSION TASKS DONE — predictions saved to volume")


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60 * 12,
)
def train_analysis():
    import shutil, os

    for variant, extra in [
        ("flash",          "--attention_type flash"),
        ("sliding_window", "--attention_type sliding_window --window_size 64"),
        ("mixed",          "--attention_type mixed --window_size 64"),
    ]:
        print(f"=== {variant} ===")
        os.makedirs("/cs224n_gpt/predictions", exist_ok=True)

        run(f"python -u sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5 {extra}")

        # Save each variant to its own subfolder so nothing gets overwritten
        dest = f"/outputs/analysis_predictions/{variant}"
        shutil.copytree("/cs224n_gpt/predictions", dest, dirs_exist_ok=True)
        volume.commit()
        print(f"{variant} done — saved to {dest}")

    print("ALL ANALYSIS TASKS DONE")


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60 * 6,
)
def train_cfimdb_ablation():
    """Sliding-window window-size ablation on CFIMDB (full-model fine-tune)."""
    import shutil, os

    for window_size in [32, 64, 128, 256]:
        print(f"=== CFIMDB sliding_window w={window_size} ===")
        os.makedirs("/cs224n_gpt/predictions", exist_ok=True)

        run(
            f"python -u classifier.py --use_gpu --fine-tune-mode full-model "
            f"--epochs 5 --lr 1e-5 "
            f"--attention_type sliding_window --window_size {window_size}"
        )

        dest = f"/outputs/ablation_cfimdb/w{window_size}"
        shutil.copytree("/cs224n_gpt/predictions", dest, dirs_exist_ok=True)
        volume.commit()
        print(f"w={window_size} done — saved to {dest}")

    print("CFIMDB ABLATION DONE")


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60 * 3,
)
def train_global_token_ablation():
    """Global token ablation: sliding window w=16 on CFIMDB with 0, 2, 4, 8 global tokens."""
    import shutil, os

    for g in [0, 2, 4, 8]:
        print(f"=== num_global_tokens={g} ===")
        os.makedirs("/cs224n_gpt/predictions", exist_ok=True)

        run(
            f"python -u classifier.py --use_gpu --fine-tune-mode full-model "
            f"--epochs 5 --lr 1e-5 "
            f"--attention_type sliding_window --window_size 16 "
            f"--num_global_tokens {g}"
        )

        dest = f"/outputs/ablation_global_w16/g{g}"
        shutil.copytree("/cs224n_gpt/predictions", dest, dirs_exist_ok=True)
        volume.commit()
        print(f"g={g} done — saved to {dest}")

    print("GLOBAL TOKEN ABLATION DONE")


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60 * 4,
)
def train_sonnet_extended():
    """Train sonnet generation for more epochs to improve ChrF score."""
    import shutil, os
    os.makedirs("/cs224n_gpt/predictions", exist_ok=True)

    run("python -u sonnet_generation.py --use_gpu --epochs 20 --lr 1e-5")

    shutil.copytree("/cs224n_gpt/predictions", "/outputs/sonnet_extended", dirs_exist_ok=True)
    volume.commit()
    print("SONNET EXTENDED TRAINING DONE")


@app.function(
    gpu="A10G",
    image=image,
    volumes={"/outputs": volume},
    timeout=60 * 60 * 2,
)
def run_benchmark():
    run("python -u benchmark.py")
    print("BENCHMARK DONE")


@app.local_entrypoint()
def main(all: bool = False):
    print("Starting submission runs...")
    train_submission.remote()

    if all:
        print("Starting analysis runs...")
        train_analysis.remote()
        print("Starting benchmark...")
        run_benchmark.remote()
