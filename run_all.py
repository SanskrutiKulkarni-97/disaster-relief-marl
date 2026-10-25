"""Run the whole experiment pipeline with one command.

    python run_all.py                  # full run: train everything, evaluate, figures, report
    python run_all.py --skip-training  # reuse the trained runs in outputs/ (takes ~2-3 min)
    python run_all.py --quick          # fast smoke test into outputs_quick/ (a few minutes)

Steps
  1. train   MAPPO original (train_mappo.py), MAPPO corrected (train_mappo_v2.py),
             Q-Learning, SARSA, DQN, PPO, masked PPO (compare_algos.py)
             -> one folder per algorithm and seed inside the output folder
  2. eval    evaluate_all.py: baselines + every trained policy on 300 held-out scenarios
             -> final_results.json
  3. figures make_figures.py -> fig_comparison.pdf, fig_learning_curves.pdf
  4. report  fill_report.py -> final_report.tex with every number filled in
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
ALGOS = ["qlearning", "sarsa", "dqn", "ppo", "mppo"]


def script(name):
    return os.path.join(ROOT, name)


def run(cmd, cwd, log_path=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg",
               PYTHONUNBUFFERED="1")
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return 0
    with open(log_path, "w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="training runs to execute in parallel")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-training", action="store_true",
                   help="reuse existing trained runs in the output folder")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 300 episodes, seed 42 only, written to outputs_quick/")
    args = p.parse_args()

    if args.quick:
        args.episodes, args.seeds = 300, [42]
    out = os.path.abspath(args.output_dir or os.path.join(ROOT, "outputs_quick" if args.quick else "outputs"))
    report_dir = out if args.quick else os.path.join(ROOT, "report")
    os.makedirs(out, exist_ok=True)
    py = sys.executable
    t0 = time.time()

    # ------------------------------------------------------------ 1. training
    if not args.skip_training:
        logs = os.path.join(out, "logs")
        os.makedirs(logs, exist_ok=True)
        jobs = []
        for s in args.seeds:
            jobs.append((f"mappo_original_s{s}", [py, script("train_mappo.py"), "--episodes", str(args.episodes),
                                                 "--seed", str(s), "--out", f"runs_v1_s{s}"]))
            jobs.append((f"mappo_corrected_s{s}", [py, script("train_mappo_v2.py"), "--episodes", str(args.episodes),
                                                  "--seed", str(s), "--out", f"runs_v2_s{s}"]))
            for a in ALGOS:
                jobs.append((f"{a}_s{s}", [py, script("compare_algos.py"), "--algo", a, "--seed", str(s),
                                           "--episodes", str(args.episodes)]))
        print(f"[1/4] training {len(jobs)} runs ({args.episodes} episodes each, {args.jobs} in parallel)")
        print(f"      logs: {logs}")
        failed = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run, cmd, out, os.path.join(logs, f"{name}.txt")): name for name, cmd in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                name = futures[fut]
                ok = fut.result() == 0
                if not ok:
                    failed.append(name)
                print(f"      [{i}/{len(jobs)}] {name} {'done' if ok else 'FAILED'}  ({(time.time() - t0) / 60:.1f} min)")
        if failed:
            sys.exit(f"Training failed for: {', '.join(failed)}. See the matching files in {logs}")
    else:
        print("[1/4] skipping training, reusing runs in", out)

    # ------------------------------------------------------------ 2-4
    print("[2/4] evaluating all policies on 300 held-out scenarios")
    run([py, script("evaluate_all.py")], out)
    print("[3/4] drawing figures")
    run([py, script("make_figures.py"), "--out-dir", report_dir], out)
    print("[4/4] filling the report")
    run([py, script("fill_report.py"), "--results", os.path.join(out, "final_results.json"),
         "--template", os.path.join(ROOT, "report", "report_template.tex"),
         "--out", os.path.join(report_dir, "final_report.tex")], ROOT)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min.")
    print(f"Report and figures: {report_dir}")


if __name__ == "__main__":
    main()
