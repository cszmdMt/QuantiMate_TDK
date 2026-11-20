"""Parallel launcher of the repeated experiment: one unchanged serial runner per (target, horizon) track.

This file is NOT part of the protocol and trains nothing itself. Every runner is the ordinary
`run_experiments.py run`, started for one version and one track in its own output directory
(the runner allows one process per output directory). The launcher only starts the six runners,
reports progress, and at the end copies the six result folders into one and rebuilds the tables.

    python experiments/runners/run_parallel.py V8        # day 1: six runners, V8 only
    python experiments/runners/run_parallel.py V8.5      # day 2: the same six folders, V8.5 only
    python experiments/runners/run_parallel.py status    # counts only, no training
    python experiments/runners/run_parallel.py merge     # six folders -> one, then the official export

Interrupted run: start the same command again; finished jobs are kept, only missing ones run.
Between the two days nothing may change in quantimate/, the generators, the notebooks or the
snapshot: the runner compares their fingerprint with the one saved on day 1 and refuses otherwise.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantimate.experiment_protocol import (DATE_PAIRS, SEEDS, TRACKS, VERSIONS,  # noqa: E402
                                            job_key, read_json, track_name)

RUNNER = REPO / "experiments/runners/run_experiments.py"
DEFAULT_ROOT = REPO / "data/test2"
COPIED = ("results", "notebooks", "jobs", "logs", "failures")  # the rest is rebuilt by export
RESTARTS = 2            # automatic restarts of a runner that stopped before its slice was complete
MIN_FREE_GB = 30


def say(text=""):
    print(text, flush=True)


def stamp():
    return datetime.now().strftime("%H:%M:%S")


def track_dirs(root):
    return {track_name(t, h): (t, h, Path(root) / "parallel" / track_name(t, h)) for t, h in TRACKS}


def result_file(directory, version, pair, seed, target, horizon):
    return directory / "results" / (job_key(version, pair, seed, target, horizon) + ".json")


def progress(directory, version, target, horizon, pairs, seeds):
    """Saved selections, saved repetitions and failure logs of one version in one track folder."""
    selections = sum(result_file(directory, version, p, None, target, horizon).exists() for p in pairs)
    runs = sum(result_file(directory, version, p, s, target, horizon).exists() for p in pairs for s in seeds)
    slice_dir = Path(job_key(version, 1, None, target, horizon)).parents[1]
    failures = len(list((directory / "failures" / slice_dir).rglob("*.jsonl")))
    return selections, runs, failures


PREFLIGHT = """
import json, sys
from quantimate.experiment_data import verify_snapshot
from quantimate.experiment_protocol import PROTOCOL
from quantimate.experiment_runner import REPO, code_fingerprint, environment_fingerprint
manifest = verify_snapshot(sys.argv[1])
frozen = dict(protocol=PROTOCOL, snapshot_id=manifest["snapshot_id"],
              code=code_fingerprint(REPO), environment=environment_fingerprint())
print(json.dumps(dict(frozen=frozen, files=len(manifest["files"]), missing=manifest["missing_optional"])))
"""


def expected_manifest(snapshot):
    """What the runner would freeze right now (same construction as prepare_experiment).

    Computed in a fresh interpreter, like the runner's own: the snapshot is verified byte by byte
    and the heavy imports (with their console banners) stay out of the launcher process."""
    done = subprocess.run([sys.executable, "-c", PREFLIGHT, str(snapshot)], cwd=REPO, env=runner_environment(),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        say(done.stderr[-3000:])
        raise SystemExit("HIBA: a pillanatkép vagy a kód ellenőrzése nem sikerült (részletek fent). Semmi nem indult el.")
    checked = json.loads(done.stdout.strip().splitlines()[-1])
    return checked["frozen"], checked


def differences(saved, now):
    changed = [key for key in ("protocol", "snapshot_id", "environment") if saved.get(key) != now.get(key)]
    a, b = saved.get("code", {}), now.get("code", {})
    return changed + ["code: " + name for name in sorted(set(a) | set(b)) if a.get(name) != b.get(name)]


def keep_awake(on):
    """Windows only: the machine must not go to sleep while the runners work."""
    if os.name != "nt":
        return
    import ctypes
    continuous, system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(continuous | (system_required if on else 0))


def free_memory_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return None


def preflight(snapshot, root):
    if not (Path(snapshot) / "snapshot.json").exists():
        raise SystemExit(f"HIBA: nincs pillanatkép itt: {snapshot}")
    now, checked = expected_manifest(snapshot)
    say(f"Pillanatkép rendben: {now['snapshot_id'][:16]}… ({checked['files']} fájl)")
    for name in checked["missing"]:
        say("FIGYELEM: ez az adatréteg hiányzik a pillanatképből: " + name)
    for name, (_, _, directory) in track_dirs(root).items():
        file = directory / "experiment.json"
        if not file.exists():
            continue
        saved = read_json(file)
        changed = differences(saved, now)
        if changed:
            say(f"HIBA: a(z) {name} mappa korábbi futása más állapotból készült. Eltérés:")
            for item in changed:
                say("   - " + item)
            raise SystemExit("A futás nem indult el. A korábbi eredmények csak az eredeti kóddal, környezettel és "
                             "pillanatképpel folytathatók; állítsd vissza az eltérő elemet (pl. git checkout).")
    free = shutil.disk_usage(Path(root).anchor or root).free / 2 ** 30
    if free < MIN_FREE_GB:
        say(f"FIGYELEM: csak {free:.0f} GB szabad hely van; a teljes kísérlet nagyjából 20–25 GB-ot kér.")


def runner_environment():
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8")
    return env


def start_runner(version, name, target, horizon, directory, snapshot, args, log_dir):
    command = [sys.executable, str(RUNNER), "run", "--snapshot", str(snapshot), "--output", str(directory),
               "--versions", version, "--targets", target, "--horizons", str(horizon)]
    if args.pairs:
        command += ["--pairs", *map(str, args.pairs)]
    if args.seeds:
        command += ["--seeds", *map(str, args.seeds)]
    env = runner_environment()
    log = (log_dir /f"{version.replace('.', '_')}_{name}.log").open("a", encoding="utf-8")
    log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} start: {' '.join(command)}\n")
    log.flush()
    return subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log


def launch(version, args):
    snapshot, root = Path(args.snapshot).resolve(), Path(args.root).resolve()
    pairs = sorted(set(args.pairs)) if args.pairs else list(range(1, len(DATE_PAIRS) + 1))
    seeds = sorted(set(args.seeds)) if args.seeds else list(SEEDS)
    need_selections, need_runs = len(pairs), len(pairs) * len(seeds)
    preflight(snapshot, root)
    log_dir = root / "parallel" / "_launcher"
    log_dir.mkdir(parents=True, exist_ok=True)

    tracks = {}
    for name, (target, horizon, directory) in track_dirs(root).items():
        selections, runs, _ = progress(directory, version, target, horizon, pairs, seeds)
        tracks[name] = dict(target=target, horizon=horizon, directory=directory, process=None, log=None,
                            starts=0, state="waiting", runs0=runs, t0=None,
                            complete=(selections == need_selections and runs == need_runs))
    for name, t in tracks.items():
        if t["complete"]:
            t["state"] = "done"
            say(f"{name}: a(z) {version} szelete már kész, nem indul újra.")
    if all(t["state"] == "done" for t in tracks.values()):
        say(f"A(z) {version} minden sávja kész. Következő lépés: merge.")
        return 0

    say(f"{stamp()}  {version}: {sum(t['state'] == 'waiting' for t in tracks.values())} sáv indul, "
        f"egyszerre legfeljebb {args.max_parallel}. Naplók: {log_dir}")
    say("Leállítás: Ctrl+C. Folytatás: ugyanez a parancs még egyszer.")
    keep_awake(True)
    began, last_report = time.time(), 0.0
    try:
        while True:
            for name, t in tracks.items():
                if t["state"] != "running" or t["process"].poll() is None:
                    continue
                t["log"].close()
                selections, runs, failures = progress(t["directory"], version, t["target"], t["horizon"], pairs, seeds)
                if selections == need_selections and runs == need_runs:
                    t["state"] = "done"
                    say(f"{stamp()}  {name}: KÉSZ ({runs}/{need_runs} ismétlés).")
                elif t["starts"] <= RESTARTS:
                    t["state"] = "waiting"
                    say(f"{stamp()}  {name}: a futtató megállt ({runs}/{need_runs}, hibanapló: {failures}), újraindítom.")
                else:
                    t["state"] = "stopped"
                    say(f"{stamp()}  {name}: a futtató {t['starts']}. alkalommal is megállt ({runs}/{need_runs}). "
                        f"Nézd meg: {log_dir}")
            running = sum(t["state"] == "running" for t in tracks.values())
            for name, t in tracks.items():
                if t["state"] == "waiting" and running < args.max_parallel:
                    t["process"], t["log"] = start_runner(version, name, t["target"], t["horizon"],
                                                          t["directory"], snapshot, args, log_dir)
                    t["starts"] += 1
                    t["state"], t["t0"] = "running", time.time()
                    t["runs0"] = progress(t["directory"], version, t["target"], t["horizon"], pairs, seeds)[1]
                    running += 1
            if not any(t["state"] in ("running", "waiting") for t in tracks.values()):
                break
            if time.time() - last_report >= (args.poll if last_report else min(args.poll, 60)):
                last_report = time.time()
                report(version, tracks, pairs, seeds, began)
            time.sleep(5)
    except KeyboardInterrupt:
        say(f"\n{stamp()}  Megszakítás: a futtatók leállnak (az éppen futó feladat elvész, a többi megmarad)…")
        deadline = time.time() + 90
        for t in tracks.values():
            if t["state"] == "running":
                try:
                    t["process"].wait(timeout=max(1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    t["process"].terminate()
                t["log"].close()
        say("Folytatás: ugyanez a parancs még egyszer.")
        return 130
    finally:
        keep_awake(False)

    report(version, tracks, pairs, seeds, began)
    missing = [name for name, t in tracks.items() if t["state"] != "done"]
    if missing:
        say(f"NEM TELJES: {', '.join(missing)}. Indítsd újra ugyanezt a parancsot; ha újra megáll, a napló megmondja, miért.")
        return 2
    say(f"{stamp()}  A(z) {version} mind a hat sávja kész. Összesítés: run_parallel.py merge")
    return 0


def report(version, tracks, pairs, seeds, began):
    need = len(pairs) * len(seeds)
    elapsed = timedelta(seconds=int(time.time() - began))
    memory = free_memory_gb()
    say(f"--- {stamp()}  {version}, eltelt: {elapsed}" + (f", szabad memória: {memory:.1f} GB" if memory is not None else ""))
    etas = []
    for name, t in tracks.items():
        selections, runs, failures = progress(t["directory"], version, t["target"], t["horizon"], pairs, seeds)
        line = f"    {name:<16} kiválasztás {selections:>2}/{len(pairs)}   ismétlés {runs:>4}/{need}   {t['state']}"
        if failures:
            line += f"   hibanapló: {failures}"
        if t["state"] == "running" and runs > t["runs0"]:
            seconds = (need - runs) * (time.time() - t["t0"]) / (runs - t["runs0"])
            etas.append(seconds)
            line += f"   hátra kb. {timedelta(seconds=int(seconds))}"
        say(line)
    if etas and all(t["state"] in ("running", "done") for t in tracks.values()) and \
            len(etas) == sum(t["state"] == "running" for t in tracks.values()):
        say(f"    várható befejezés: {(datetime.now() + timedelta(seconds=max(etas))).strftime('%m-%d %H:%M')}")


def status(args):
    root = Path(args.root).resolve()
    pairs, seeds = list(range(1, len(DATE_PAIRS) + 1)), list(SEEDS)
    need = len(pairs) * len(seeds)
    identities = set()
    for version in VERSIONS:
        say(version)
        for name, (target, horizon, directory) in track_dirs(root).items():
            selections, runs, failures = progress(directory, version, target, horizon, pairs, seeds)
            say(f"    {name:<16} kiválasztás {selections:>2}/{len(pairs)}   ismétlés {runs:>4}/{need}"
                + (f"   hibanapló: {failures}" if failures else ""))
            if (directory / "experiment.json").exists():
                identities.add(read_json(directory / "experiment.json")["experiment_id"])
    if len(identities) > 1:
        say("FIGYELEM: a sávmappák kísérletazonosítója eltér — nem fésülhetők össze.")
    merged = root / "repeated_experiment" / "status.json"
    if merged.exists():
        s = read_json(merged)
        say(f"Összefésült mappa: {s['successful_runs']}/{s['required_runs']} ismétlés, állapot: {s['status']}")
    return 0


def copy_changed(source, target):
    s = os.stat(source)
    try:
        t = os.stat(target)
        if t.st_size == s.st_size and int(t.st_mtime) == int(s.st_mtime):
            return target
    except FileNotFoundError:
        pass
    return shutil.copy2(source, target)


def merge(args):
    """Six track folders -> data/.../repeated_experiment, then the runner's own `export`."""
    from quantimate.experiment_runner import experiment_lock
    root = Path(args.root).resolve()
    sources = {name: d for name, (_, _, d) in track_dirs(root).items() if (d / "experiment.json").exists()}
    if not sources:
        raise SystemExit(f"HIBA: nincs egyetlen sávmappa sem itt: {root / 'parallel'}")
    manifests = {name: read_json(d / "experiment.json") for name, d in sources.items()}
    if len({m["experiment_id"] for m in manifests.values()}) != 1:
        for name, m in manifests.items():
            say(f"    {name}: {m['experiment_id']}")
        raise SystemExit("HIBA: a sávmappák kísérletazonosítója eltér; ezek nem ugyanannak a kísérletnek a részei.")
    manifest = next(iter(manifests.values()))
    target = root / "repeated_experiment"
    if (target / "experiment.json").exists() and read_json(target / "experiment.json") != manifest:
        raise SystemExit(f"HIBA: a célmappa egy másik kísérleté: {target}")
    with ExitStack() as stack:
        for name, directory in sources.items():
            try:
                stack.enter_context(experiment_lock(directory))
            except (RuntimeError, OSError):
                raise SystemExit(f"HIBA: a(z) {name} sávon még fut egy futtató; az összefésülés csak utána indítható.")
        target.mkdir(parents=True, exist_ok=True)
        if not (target / "experiment.json").exists():
            shutil.copyfile(next(iter(sources.values())) / "experiment.json", target / "experiment.json")
        for name, directory in sources.items():
            say(f"{stamp()}  másolás: {name}")
            for part in COPIED:
                if (directory / part).exists():
                    shutil.copytree(directory / part, target / part, dirs_exist_ok=True, copy_function=copy_changed)
    say(f"{stamp()}  összesítő táblák építése (a teljes kísérletnél ez több percig tart)…")
    env = runner_environment()
    done = subprocess.run([sys.executable, str(RUNNER), "export", "--output", str(target)],
                          cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        say(done.stdout + done.stderr)
        raise SystemExit("HIBA: az összesítés nem sikerült (a sávmappák érintetlenek).")
    result = read_json(target / "status.json")
    for key, count in result["tracks"].items():
        say(f"    {key:<24} {count:>4}/{len(DATE_PAIRS) * len(SEEDS)}")
    say(f"{stamp()}  Összesen {result['successful_runs']}/{result['required_runs']} ismétlés, állapot: {result['status']}")
    say(f"Eredmények: {target}  (results.csv, tracks.csv)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Six parallel serial runners, one per track; see the module docstring.")
    parser.add_argument("command", choices=[*VERSIONS, "status", "merge"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Holds parallel/<track>/ (runner outputs) and repeated_experiment/ (merged)")
    parser.add_argument("--snapshot", type=Path, help="Default: <root>/experiment_snapshot")
    parser.add_argument("--max-parallel", type=int, default=len(TRACKS), help="Runners at the same time (default 6)")
    parser.add_argument("--poll", type=int, default=300, help="Seconds between progress reports")
    parser.add_argument("--pairs", type=int, nargs="+", choices=range(1, len(DATE_PAIRS) + 1), metavar="PAIR",
                        help="TRIAL ONLY: subset of date pairs")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, metavar="SEED",
                        help="TRIAL ONLY: subset of seeds")
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    if args.snapshot is None:
        args.snapshot = Path(args.root) / "experiment_snapshot"
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if args.command == "status":
        return status(args)
    if args.command == "merge":
        return merge(args)
    return launch(args.command, args)


if __name__ == "__main__":
    raise SystemExit(main())
