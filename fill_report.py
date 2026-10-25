"""Write report/final_report.tex by filling every <<placeholder>> in
report/report_template.tex with numbers from outputs/final_results.json.

The prose in the template was written for the full 4000-episode, 3-seed
results. If you regenerate with different settings (for example --quick),
the numbers update automatically but the script warns you when a sentence
in the text no longer matches the data.
"""
import argparse
import json
import re

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--results", default="outputs/final_results.json")
p.add_argument("--template", default="report/report_template.tex")
p.add_argument("--out", default="report/final_report.tex")
args = p.parse_args()

R = json.load(open(args.results))
C = R["comparison"]
S = {}
warnings = []


def f1(x):
    return f"$-${-x:.1f}" if x < 0 else f"{x:.1f}"


def row(tag, d):
    S[f"{tag}_r"] = f1(d["reward"]); S[f"{tag}_u"] = f1(d["crit_unmet"])
    S[f"{tag}_s"] = f"{d['served']:.0f}"; S[f"{tag}_e"] = f"{d['energy']:.0f}"
    S[f"{tag}_c"] = f"{d['completion']:.3f}"; S[f"{tag}_l"] = f1(d["length"])
    if "reward_std" in d:
        S[f"{tag}_r_std"] = f1(d["reward_std"]); S[f"{tag}_u_std"] = f1(d["crit_unmet_std"])


for tag, d in [("noop", R["noop"]), ("rnd", R["random"]), ("grd", R["greedy_heuristic"]),
               ("v1s", R["mappo_v1_stochastic"]), ("v1a", R["mappo_v1_argmax"]),
               ("v2s", R["mappo_v2_stochastic"]), ("v2a", R["mappo_v2_argmax"]),
               ("ql", C["qlearning_greedy"]), ("sa", C["sarsa_greedy"]), ("dqn", C["dqn_greedy"]),
               ("ppos", C["ppo_stochastic"]), ("ppoa", C["ppo_greedy"]),
               ("mps", C["mppo_stochastic"]), ("mpa", C["mppo_greedy"])]:
    row(tag, d)


def conv(key):
    v = C[key]
    if any(x is None for x in v):
        warnings.append(f"{key}: some seeds never reached the -50 threshold")
        reached = [x for x in v if x is not None]
        return float(np.mean(reached)) if reached else None
    return float(np.mean(v))


cv = {"dqn_cv": conv("dqn_conv"), "ppo_cv": conv("ppo_conv"),
      "mpp_cv": conv("mppo_conv"), "v2_cv": conv("mappo_v2_conv")}
S.update({k: ("n/r" if v is None else f"{v:.0f}") for k, v in cv.items()})
S["mask_speedup"] = (f"{cv['ppo_cv'] / cv['mpp_cv']:.1f}"
                     if cv["ppo_cv"] and cv["mpp_cv"] else "n/a")
S["std_ratio"] = f"{C['mppo_stochastic']['reward_std'] / max(R['mappo_v2_stochastic']['reward_std'], 1e-9):.0f}"
S["ql_states"] = f"{round(C['qlearning_states'], -2):,.0f}".replace(",", "{,}")
S["ql_unseen"] = f"{100 * C['qlearning_unseen']:.0f}"

g = R["grad_diag_v1"]
S["shrink"] = str(int(round(g["shrink_factor"], -1)))
S["ratio"] = str(int(round(g["ratio"], -1)))
S["closs"] = f"{g['critic_loss']:.0f}"
S["gc"] = f"{g['critic_grad_pre']:.0f}"
S["ga"] = f"{g['actor_grad_pre']:.2f}"
S["gpost"] = f"{g['actor_grad_post_clip']:.4f}"
S["init_demand"] = f"{R['capacity']['initial_demand_mean']:.0f}"
allc = [d["completion"] for d in [R["noop"], R["random"], R["greedy_heuristic"],
        R["mappo_v1_stochastic"], R["mappo_v1_argmax"], R["mappo_v2_stochastic"],
        R["mappo_v2_argmax"]] + [v for v in C.values() if isinstance(v, dict)]]
if max(allc) >= 0.05:
    warnings.append("a completion rate is >= 0.05; the text says all stay below 0.05")
S["max_completion"] = "0.05"
S["v2_vs_rnd_pct"] = f"{100 * (R['random']['crit_unmet'] - R['mappo_v2_stochastic']['crit_unmet']) / R['random']['crit_unmet']:.0f}"

order = sorted([("MAPPO", R["mappo_v2_stochastic"]["reward"]), ("masked PPO", C["mppo_stochastic"]["reward"]),
                ("PPO", C["ppo_stochastic"]["reward"]), ("DQN", C["dqn_greedy"]["reward"]),
                ("random", R["random"]["reward"]), ("Q-Learning", C["qlearning_greedy"]["reward"]),
                ("SARSA", C["sarsa_greedy"]["reward"])], key=lambda x: -x[1])
expected = ["MAPPO", "masked PPO", "PPO", "DQN", "random", "Q-Learning", "SARSA"]
if [n for n, _ in order] != expected:
    warnings.append("reward ranking differs from the order written in Section 'Overall Comparison': "
                    + " > ".join(n for n, _ in order))

t = open(args.template, encoding="utf-8").read()
for _ in range(3):
    t = re.sub(r"<<(\w+)>>", lambda m: S[m.group(1)], t)
assert "<<" not in t, "unfilled placeholder left in the report"
open(args.out, "w", encoding="utf-8").write(t)
print(f"wrote {args.out}")
for w in warnings:
    print("WARNING (check the report text):", w)
