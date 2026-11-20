"""Serial notebook orchestration with fresh processes/kernels and durable resume."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

from quantimate.experiment_protocol import (DATE_PAIRS, HORIZONS, PROTOCOL, REQUIRED_RUNS, SEEDS,
    TARGETS, VERSIONS, atomic_json, count_runs, digest, export_results, file_hash, job_key,
    read_json, result_valid, status_of)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "V8": "experiments/notebooks/active/mainV8.ipynb",
    "V8.5": "experiments/notebooks/active/mainV8_5.ipynb",
}


def code_fingerprint(repo):
    repo = Path(repo)
    paths = sorted((repo / "quantimate").rglob("*.py"))
    paths += [
        repo / "experiments/runners/run_experiments.py",
        repo / "experiments/generators/nb_walkforward.py",
        repo / "experiments/generators/build_notebook_v8.py",
        repo / "experiments/generators/build_notebook_v8_5.py",
    ]
    hashes = {p.relative_to(repo).as_posix(): file_hash(p) for p in paths}
    for name in NOTEBOOKS.values():
        notebook = read_json(repo / name)
        hashes[name] = digest([{k: cell[k] for k in ("cell_type", "source")} for cell in notebook["cells"]])
    return hashes


def environment_fingerprint():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "lightgbm",
                "pyarrow", "nbclient", "nbformat", "ipykernel")
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                packages={p: importlib.metadata.version(p) for p in packages})


@contextmanager
def experiment_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+b") as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another runner holds this experiment lock") from exc
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def prepare_experiment(snapshot_path, output):
    from quantimate.experiment_data import Snapshot
    reader = Snapshot(snapshot_path)
    snapshot = reader.manifest
    # Every track needs an observed target h sessions after every cutoff.
    try:
        for cutoff, _ in DATE_PAIRS:
            reader.target_date(cutoff, max(HORIZONS))
    except ValueError as exc:
        raise ValueError(f"{exc}. This snapshot is too short for the longer horizons:"
                         " create a NEW snapshot. Nothing was trained.") from exc
    for name in snapshot["missing_optional"]:
        print("WARNING: data layer absent from the snapshot: " + name, flush=True)
    frozen = dict(protocol=PROTOCOL, snapshot_id=snapshot["snapshot_id"],
                  code=code_fingerprint(REPO), environment=environment_fingerprint())
    frozen = json.loads(json.dumps(frozen))  # canonical tuples -> lists
    frozen["experiment_id"] = digest(frozen)
    file = Path(output) / "experiment.json"
    if file.exists():
        if read_json(file) != frozen:
            raise ValueError("Protocol, source, environment or snapshot changed. Use a NEW output directory.")
    else:
        if any((Path(output) / "results").rglob("*.json")):
            raise ValueError("Results exist without an experiment manifest")
        atomic_json(file, frozen)
    return frozen


def worker(job_path):
    """One OS process and one fresh kernel for exactly one notebook job."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    job_path = Path(job_path).resolve()
    job = read_json(job_path)
    notebook = nbformat.read(REPO / NOTEBOOKS[job["version"]], as_version=4)
    # Never depend on a user's existing kernel registration or interpreter.
    with tempfile.TemporaryDirectory(prefix="quantimate-kernel-") as temp:
        kernel_root = Path(temp) / "kernels"
        kernel_dir = kernel_root / "quantimate-experiment"
        kernel_dir.mkdir(parents=True)
        atomic_json(kernel_dir / "kernel.json", dict(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="QuantiMate isolated experiment", language="python",
            env={"QUANTIMATE_JOB": str(job_path), "PYTHONHASHSEED": "42",
                 "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                 "NUMEXPR_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}))
        manager = KernelManager(kernel_name="quantimate-experiment",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
        client = NotebookClient(notebook, km=manager, timeout=None,
                                resources={"metadata": {"path": str(REPO)}},
                                allow_errors=False)
        # No write to either source notebook, even on failure.
        try:
            client.execute()
        finally:
            target = Path(job["executed_notebook"])
            target.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, target)


def execute_with_retries(job, launcher=None):
    """Three attempts per invocation; same immutable job/seed on every retry."""
    output = Path(job["output"])
    key, identity = job["job_key"], job["experiment_id"]
    if result_valid(job["result_path"], identity, key):
        return True
    job_path = output / "jobs" / (key + ".json")
    atomic_json(job_path, job)
    log_path = output / "logs" / (key + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        print(f"{key}: attempt {attempt}/3", flush=True)
        try:
            if launcher is not None:
                launcher(job_path)
            else:
                env = os.environ.copy()
                env.update(PYTHONIOENCODING="utf-8", PYTHONHASHSEED="42",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n{datetime.now(timezone.utc).isoformat()} attempt={attempt}\n")
                    log.flush()
                    subprocess.run([sys.executable, str(REPO / "experiments/runners/run_experiments.py"),
                                    "_worker", str(job_path)], cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            if not result_valid(job["result_path"], identity, key):
                raise RuntimeError("Notebook finished without an atomic success result")
            return True
        except Exception as exc:
            # A completed result remains valid even if the worker failed while saving its notebook.
            if result_valid(job["result_path"], identity, key):
                return True
            error = dict(job_key=key, seed=job["seed"], attempt=attempt,
                         time_utc=datetime.now(timezone.utc).isoformat(),
                         status="failed", error=str(exc), traceback=traceback.format_exc())
            error_path = output / "failures" / (key + ".jsonl")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with error_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(error, ensure_ascii=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
    return False


def run(snapshot_path, output, versions=VERSIONS, pair_ids=None, max_jobs=None,
        targets=TARGETS, horizons=HORIZONS, seeds=SEEDS):
    snapshot_path, output = Path(snapshot_path).resolve(), Path(output).resolve()
    if output == snapshot_path or output.is_relative_to(snapshot_path):
        raise ValueError("Output directory must be outside the immutable snapshot")
    completed_now = 0
    wanted_pairs = set(pair_ids if pair_ids is not None else range(1, 31))
    with experiment_lock(output):
        manifest = prepare_experiment(snapshot_path, output)
        try:
            tracks = [(version, target, horizon) for version in versions
                      for target in targets for horizon in horizons]
            for version, target, horizon in tracks:
                for pair_id in sorted(wanted_pairs):
                    selection_key = job_key(version, pair_id, None, target, horizon)
                    selection_path = output / "results" / (selection_key + ".json")
                    def make_job(seed):
                        key = job_key(version, pair_id, seed, target, horizon)
                        return dict(experiment_id=manifest["experiment_id"],
                                    output=str(output), snapshot=str(snapshot_path),
                                    version=version, pair_id=pair_id, target=target,
                                    horizon=horizon, seed=42 if seed is None else seed,
                                    phase="select" if seed is None else "repeat", job_key=key,
                                    result_path=str(output / "results" / (key + ".json")),
                                    executed_notebook=str(output / "notebooks" / (key + ".ipynb")))
                    if not selection_path.exists() and any(selection_path.parent.glob("seed_*.json")):
                        raise ValueError("Saved repetitions exist but their frozen selection is missing; restore the selection file")
                    if not execute_with_retries(make_job(None)):
                        continue
                    selection_hash = read_json(selection_path)["result_hash"]
                    for seed in sorted(seeds):
                        job = make_job(seed)
                        job.update(selection_path=str(selection_path), selection_hash=selection_hash)
                        if result_valid(job["result_path"], manifest["experiment_id"], job["job_key"]):
                            if read_json(job["result_path"])["selection_hash"] != selection_hash:
                                raise ValueError("Saved repetition belongs to a different frozen selection")
                            continue
                        if max_jobs is not None and completed_now >= max_jobs:
                            return export_results(output)
                        execute_with_retries(job)
                        completed_now += 1
        finally:
            status = export_results(output)
            print(json.dumps(status), flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="V8/V8.5: 30 date pairs x 50 independent repetitions per (target, horizon) track. No default execution.")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Print protocol only; no download/training")
    snap = commands.add_parser("snapshot", help="Create immutable data snapshot; no training")
    snap.add_argument("--destination", type=Path, required=True)
    snap.add_argument("--cache", type=Path, default=REPO / "data")
    snap.add_argument("--download", action="store_true", help="Explicitly download daily history since 2001-02-01")
    snap.add_argument("--allow-missing", nargs="+", default=[], metavar="SOURCE",
                      help="Optional sources whose absence is accepted; any other missing source is an error")
    launch = commands.add_parser("run", help="EXECUTE notebooks and train models, serially")
    launch.add_argument("--snapshot", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--versions", choices=VERSIONS, nargs="+", default=list(VERSIONS))
    launch.add_argument("--pairs", type=int, nargs="+", choices=range(1, 31))
    launch.add_argument("--targets", choices=TARGETS, nargs="+", default=list(TARGETS))
    launch.add_argument("--horizons", type=int, choices=HORIZONS, nargs="+", default=list(HORIZONS))
    launch.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, metavar="SEED",
                        help="Subset of the protocol seeds 1-50; the completion target is unchanged")
    launch.add_argument("--max-jobs", type=int,
                        help=f"Optional repetition-job limit; completion target remains {REQUIRED_RUNS}")
    for name in ("status", "export"):
        sub = commands.add_parser(name, help="Read saved results; never train")
        sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("_worker", help=argparse.SUPPRESS)
    sub.add_argument("job", type=Path)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "plan":
        print(json.dumps(PROTOCOL, indent=2))
    elif args.command == "snapshot":
        from quantimate.experiment_data import create_snapshot
        m = create_snapshot(args.destination, args.cache, download=args.download,
                            allow_missing=args.allow_missing)
        print("Snapshot created: " + m["snapshot_id"])
    elif args.command == "run":
        if args.max_jobs is not None and args.max_jobs < 1:
            parser.error("--max-jobs must be positive")
        unique = lambda values: tuple(dict.fromkeys(values))
        status = run(args.snapshot, args.output, unique(args.versions), args.pairs, args.max_jobs,
                     unique(args.targets), unique(args.horizons),
                     unique(args.seeds) if args.seeds else SEEDS)
        return 0 if status["status"] == "complete" else 2
    elif args.command == "export":
        with experiment_lock(args.output):
            print(json.dumps(export_results(args.output)))
    elif args.command == "status":
        manifest = read_json(args.output / "experiment.json")
        print(json.dumps(status_of(count_runs(args.output, manifest["experiment_id"]))))
    else:
        worker(args.job)
    return 0
