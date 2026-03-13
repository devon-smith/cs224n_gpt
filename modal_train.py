"""
Modal training script for CS 224N GPT-2 experiments.

Usage:
  modal run --detach modal_train.py::run_all
  modal run --detach modal_train.py::run_phase --phase 4
  modal run --detach modal_train.py::run_phases --start 4 --end 10

Phases:
  1 = Baseline classifiers (SST + CFIMDB)
  2 = Baseline paraphrase detection
  3 = Baseline sonnet generation
  4 = Sliding window (all tasks)
  5 = Flash attention (all tasks)
  6 = Window size ablation (SST)
  7 = Efficiency benchmark
  8 = Global token ablation (SST)
  9 = Global tokens: paraphrase + sonnet (sliding_window + 2 global tokens)
  10 = ChrF scoring
  11 = Results summary
"""

import modal

app = modal.App("cs224n-gpt2")

# Persistent volume for checkpoints, predictions, and logs
vol = modal.Volume.from_name("cs224n-results", create_if_missing=True)

# Build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sacrebleu",
        "scikit-learn",
        "tqdm",
        "numpy",
        "requests",
        "importlib_metadata",
        "einops",
    )
    .add_local_dir(".", remote_path="/app", copy=True, ignore=[
        "__pycache__",
        "*.pt",
        ".git",
        ".claude",
        "training_log.txt",
        "predictions/run1",
        "predictions/run2",
        "predictions/.DS_Store",
        ".DS_Store",
    ])
)

GPU_TYPE = "A100-80GB"
TIMEOUT = 28800  # 8 hours per function

LOG_FILE = "/app/training_log.txt"


def _setup():
    """Create required directories and set env vars."""
    import os
    os.makedirs("/app/predictions", exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _run_cmd(cmd: str, phase_header: str = ""):
    """Run a shell command, stream output, and append to training_log.txt."""
    import subprocess
    print(f"\n>>> {cmd}", flush=True)

    # Write phase header to log if provided
    if phase_header:
        with open(LOG_FILE, "a") as log:
            log.write(f"{phase_header}\n")

    process = subprocess.Popen(
        cmd, shell=True, cwd="/app",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    output_lines = []
    with open(LOG_FILE, "a") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            output_lines.append(line)
    process.wait()
    output = "".join(output_lines)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (exit {process.returncode}): {cmd}")
    return output


def _sync_outputs():
    """Copy results from /app to the persistent volume."""
    import shutil
    import os
    vol_root = "/results"
    os.makedirs(vol_root, exist_ok=True)

    # Copy checkpoint files, training log, and CSV files
    for f in os.listdir("/app"):
        if f.endswith(".pt") or f == "training_log.txt" or f.endswith(".csv"):
            shutil.copy2(f"/app/{f}", f"{vol_root}/{f}")

    # Copy predictions (files only, skip subdirectories)
    pred_src = "/app/predictions"
    pred_dst = f"{vol_root}/predictions"
    if os.path.isdir(pred_src):
        os.makedirs(pred_dst, exist_ok=True)
        for f in os.listdir(pred_src):
            src_path = f"{pred_src}/{f}"
            if os.path.isfile(src_path):
                shutil.copy2(src_path, f"{pred_dst}/{f}")

    modal.Volume.from_name("cs224n-results").commit()


def _restore_checkpoints():
    """Restore any previously saved checkpoints from volume."""
    import shutil
    import os
    vol_root = "/results"
    if not os.path.isdir(vol_root):
        return

    for f in os.listdir(vol_root):
        src = f"{vol_root}/{f}"
        if os.path.isfile(src) and f.endswith(".pt"):
            shutil.copy2(src, f"/app/{f}")

    pred_src = f"{vol_root}/predictions"
    if os.path.isdir(pred_src):
        os.makedirs("/app/predictions", exist_ok=True)
        for f in os.listdir(pred_src):
            shutil.copy2(f"{pred_src}/{f}", f"/app/predictions/{f}")


def _clear_volume():
    """Wipe all previous results from the persistent volume for a clean run."""
    import shutil
    import os
    vol_root = "/results"
    if os.path.isdir(vol_root):
        for f in os.listdir(vol_root):
            path = os.path.join(vol_root, f)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        print("Cleared previous results from volume.")
    # Also clear any local log so we don't append to stale data
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    modal.Volume.from_name("cs224n-results").commit()


def _phase_1():
    """Baseline classifiers."""
    _run_cmd(
        "python classifier.py --use_gpu --fine-tune-mode last-linear-layer"
        " --epochs 10 --lr 1e-3",
        phase_header="=== Phase 1: Sentiment (last-linear-layer) ==="
    )
    _run_cmd(
        "python classifier.py --use_gpu --fine-tune-mode full-model"
        " --epochs 5 --lr 1e-5",
        phase_header="=== Phase 1: Sentiment (full-model) ==="
    )


def _phase_2():
    """Baseline paraphrase detection."""
    _run_cmd(
        "python paraphrase_detection.py --use_gpu --epochs 5 --lr 1e-5"
        " --batch_size 64",
        phase_header="=== Phase 2: Paraphrase Detection ==="
    )


def _phase_3():
    """Baseline sonnet generation."""
    _run_cmd(
        "python sonnet_generation.py --use_gpu --epochs 5 --lr 1e-5",
        phase_header="=== Phase 3: Baseline Sonnet Generation ==="
    )
    # Also generate dev sonnets from the just-trained checkpoint
    _run_cmd("""python -c "
from sonnet_generation import get_args, add_arguments, generate_submission_sonnets, seed_everything
args = get_args()
args.filepath = '5-1e-05-sonnet.pt'
args.epochs = 5
args.use_gpu = True
add_arguments(args)
seed_everything(args.seed)
args.held_out_sonnet_path = 'data/sonnets_held_out_dev.txt'
args.sonnet_out = 'predictions/generated_sonnets_dev.txt'
generate_submission_sonnets(args)
"
""", phase_header="=== Phase 3: Baseline Sonnet Dev Generation ===")


def _phase_4():
    """Sliding window attention experiments."""
    _run_cmd(
        "python classifier.py --use_gpu --fine-tune-mode full-model"
        " --epochs 5 --lr 1e-5 --attention_type sliding_window --window_size 128",
        phase_header="=== Phase 4: Sliding Window: Sentiment (full-model) ==="
    )
    _run_cmd(
        "python paraphrase_detection.py --use_gpu --epochs 5 --lr 1e-5"
        " --batch_size 64"
        " --attention_type sliding_window --window_size 128",
        phase_header="=== Phase 4: Sliding Window: Paraphrase Detection ==="
    )
    _run_cmd(
        "python sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5"
        " --attention_type sliding_window --window_size 128",
        phase_header="=== Phase 4: Sliding Window: Sonnet Generation ==="
    )


def _phase_5():
    """Flash attention experiments."""
    _run_cmd(
        "python classifier.py --use_gpu --fine-tune-mode full-model"
        " --epochs 5 --lr 1e-5 --attention_type flash",
        phase_header="=== Phase 5: Flash Attention: Sentiment (full-model) ==="
    )
    _run_cmd(
        "python paraphrase_detection.py --use_gpu --epochs 5 --lr 1e-5"
        " --batch_size 64"
        " --attention_type flash",
        phase_header="=== Phase 5: Flash Attention: Paraphrase Detection ==="
    )
    _run_cmd(
        "python sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5"
        " --attention_type flash",
        phase_header="=== Phase 5: Flash Attention: Sonnet Generation ==="
    )


def _phase_6():
    """Window size ablation (SST only)."""
    for ws in [64, 128, 256, 512]:
        _run_cmd(
            f"python classifier.py --use_gpu --fine-tune-mode full-model"
            f" --epochs 3 --lr 1e-5 --attention_type sliding_window --window_size {ws}",
            phase_header=f"--- window_size={ws} ---"
        )


def _phase_7():
    """Efficiency benchmark."""
    _run_cmd(
        "python benchmark.py",
        phase_header="=== Phase 7: Efficiency Benchmark ==="
    )


def _phase_8():
    """Global token ablation (SST only)."""
    for ng in [0, 1, 2, 4, 8, 16]:
        _run_cmd(
            f"python classifier.py --use_gpu --fine-tune-mode full-model"
            f" --epochs 5 --lr 1e-5 --attention_type sliding_window"
            f" --window_size 128 --num_global_tokens {ng}",
            phase_header=f"=== Phase 8: Global Token Ablation (g={ng}) ==="
        )


def _phase_9():
    """Global tokens: paraphrase + sonnet (sliding_window + 2 global tokens)."""
    _run_cmd(
        "python paraphrase_detection.py --use_gpu --epochs 5 --lr 1e-5"
        " --batch_size 64"
        " --attention_type sliding_window --window_size 128 --num_global_tokens 2",
        phase_header="=== Phase 9: Global Tokens: Paraphrase Detection ==="
    )
    _run_cmd(
        "python sonnet_generation.py --use_gpu --epochs 10 --lr 1e-5"
        " --attention_type sliding_window --window_size 128 --num_global_tokens 2",
        phase_header="=== Phase 9: Global Tokens: Sonnet Generation ==="
    )


def _phase_10():
    """ChrF scoring."""
    _run_cmd("""python -c "
from evaluation import test_sonnet
gold = 'data/TRUE_sonnets_held_out_dev.txt'
for name, path in [
    ('full', 'predictions/generated_sonnets.txt'),
    ('sliding_window', 'predictions/generated_sonnets-sliding_window-ws128.txt'),
    ('flash', 'predictions/generated_sonnets-flash.txt'),
    ('sw_global2', 'predictions/generated_sonnets-sliding_window-ws128-g2.txt'),
]:
    try:
        score = test_sonnet(test_path=path, gold_path=gold)
        print(f'{name}: ChrF = {score:.2f}')
    except Exception as e:
        print(f'{name}: not available ({e})')
"
""", phase_header="=== Phase 10: ChrF Scoring ===")


def _phase_11():
    """Results summary."""
    _run_cmd(
        "python results_summary.py --log training_log.txt --benchmark benchmark_results.csv",
        phase_header="=== Phase 11: Results Summary ==="
    )


PHASES = {
    1: ("Baseline Classifiers", _phase_1),
    2: ("Baseline Paraphrase", _phase_2),
    3: ("Baseline Sonnet", _phase_3),
    4: ("Sliding Window", _phase_4),
    5: ("Flash Attention", _phase_5),
    6: ("Window Ablation", _phase_6),
    7: ("Benchmark", _phase_7),
    8: ("Global Token Ablation", _phase_8),
    9: ("Global Tokens: Para + Sonnet", _phase_9),
    10: ("ChrF Scoring", _phase_10),
    11: ("Results Summary", _phase_11),
}


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT,
    volumes={"/results": vol},
)
def run_phase(phase: int):
    """Run a single experiment phase."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase {phase}. Valid: {list(PHASES.keys())}")

    _setup()
    _restore_checkpoints()

    name, fn = PHASES[phase]
    print(f"\n{'='*60}")
    print(f"Phase {phase}: {name}")
    print(f"{'='*60}", flush=True)
    fn()
    _sync_outputs()
    print(f"\nPhase {phase} complete.")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,  # 24 hours for full run
    volumes={"/results": vol},
)
def run_all():
    """Run all experiment phases sequentially (clean start)."""
    _setup()
    _clear_volume()

    for phase_num in sorted(PHASES.keys()):
        name, fn = PHASES[phase_num]
        print(f"\n{'='*60}")
        print(f"Phase {phase_num}: {name}")
        print(f"{'='*60}", flush=True)
        fn()
        _sync_outputs()

    print("\n" + "="*60)
    print("ALL PHASES COMPLETE")
    print("="*60)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=86400,  # 24 hours for multi-phase runs
    volumes={"/results": vol},
)
def run_phases(start: int = 1, end: int = 11, clean: bool = True):
    """Run a range of phases (inclusive). Set clean=False to keep prior results."""
    _setup()
    if clean:
        _clear_volume()
    else:
        _restore_checkpoints()

    for phase_num in range(start, end + 1):
        if phase_num not in PHASES:
            continue
        name, fn = PHASES[phase_num]
        print(f"\n{'='*60}")
        print(f"Phase {phase_num}: {name}")
        print(f"{'='*60}", flush=True)
        fn()
        _sync_outputs()

    print("\n" + "="*60)
    print(f"PHASES {start}-{end} COMPLETE")
    print("="*60)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT,
    volumes={"/results": vol},
)
def regenerate_sonnets(epoch: int = 4, temperature: float = 1.2):
    """Regenerate sonnets from an existing checkpoint without retraining."""
    _setup()
    _restore_checkpoints()

    total_epochs = 10
    ckpt = f"{epoch}_{total_epochs}-1e-05-sonnet.pt"

    # Generate test sonnets
    _run_cmd(f"""python -c "
from sonnet_generation import get_args, add_arguments, generate_submission_sonnets, seed_everything
args = get_args()
args.filepath = '{total_epochs}-1e-05-sonnet.pt'
args.epochs = {epoch + 1}
args.temperature = {temperature}
args.use_gpu = True
add_arguments(args)
seed_everything(args.seed)
generate_submission_sonnets(args)
" """, phase_header=f"=== Regenerate: Test Sonnets (epoch={epoch}, temp={temperature}) ===")

    # Generate dev sonnets
    _run_cmd(f"""python -c "
from sonnet_generation import get_args, add_arguments, generate_submission_sonnets, seed_everything
args = get_args()
args.filepath = '{total_epochs}-1e-05-sonnet.pt'
args.epochs = {epoch + 1}
args.temperature = {temperature}
args.use_gpu = True
add_arguments(args)
seed_everything(args.seed)
args.held_out_sonnet_path = 'data/sonnets_held_out_dev.txt'
args.sonnet_out = 'predictions/generated_sonnets_dev.txt'
generate_submission_sonnets(args)
" """, phase_header=f"=== Regenerate: Dev Sonnets (epoch={epoch}, temp={temperature}) ===")

    _sync_outputs()
    print(f"Done. Generated from checkpoint {ckpt} with temp={temperature}")


@app.function(
    image=image,
    volumes={"/results": vol},
)
def clear_results():
    """Wipe all data from the persistent volume."""
    _clear_volume()
    print("Volume cleared. Ready for a fresh run.")


@app.function(
    image=image,
    volumes={"/results": vol},
)
def download_results():
    """Print all results from the volume."""
    import os

    vol_root = "/results"
    print("=== Files in results volume ===")
    for root, dirs, files in os.walk(vol_root):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {path} ({size:,} bytes)")

    # Print training log
    log_path = f"{vol_root}/training_log.txt"
    if os.path.exists(log_path):
        print(f"\n{'='*60}")
        print("TRAINING LOG")
        print(f"{'='*60}")
        with open(log_path) as f:
            print(f.read())

    # Print benchmark CSV
    csv_path = f"{vol_root}/benchmark_results.csv"
    if os.path.exists(csv_path):
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        with open(csv_path) as f:
            print(f.read())

    # Print generated sonnets
    pred_dir = f"{vol_root}/predictions"
    if os.path.isdir(pred_dir):
        for f in sorted(os.listdir(pred_dir)):
            if f.startswith("generated_sonnets"):
                path = os.path.join(pred_dir, f)
                print(f"\n{'='*60}")
                print(f"SONNETS: {f}")
                print(f"{'='*60}")
                with open(path) as fh:
                    print(fh.read()[:3000])
