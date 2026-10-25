# AI Disaster Relief Supply Chain Optimization

Multi-agent reinforcement learning for disaster-relief supply allocation:
three warehouses dispatch water, food and medicine to five affected zones
under stochastic demand, road failures, limited vehicles and humanitarian
priorities. The project compares Q-Learning, SARSA, DQN, PPO, action-masked
PPO and MAPPO on the same environment and reward, and contains the code,
the trained results and the final IEEE report.

## Setup (VS Code, Windows / macOS / Linux)

1. Open this folder in VS Code (**File > Open Folder**).
2. Create a virtual environment and install the requirements in the VS Code terminal:

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Select the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

## Run

Use the **Run and Debug** panel (Ctrl+Shift+D) and pick one of the three
configurations, or run the same commands in the terminal:

| What | Command | Time |
|---|---|---|
| Rebuild evaluation, figures and report from the included trained runs | `python run_all.py --skip-training` | ~1 min |
| Quick smoke test of the whole pipeline (300 episodes, 1 seed, written to `outputs_quick/`) | `python run_all.py --quick` | ~1-3 min |
| Full experiment from scratch (7 methods x 3 seeds x 4000 episodes) | `python run_all.py` | ~30-60 min, depends on CPU cores |

The full run overwrites `outputs/` and `report/`. Use `--jobs N` to set how
many training runs execute in parallel (default: CPU cores minus one).

## Files

| File | Purpose |
|---|---|
| `disaster_env.py` | The simulator: warehouses, zones, demand, roads, vehicles, reward, action masks |
| `train_mappo.py` | Original MAPPO implementation (kept unchanged; it does not learn, see report Section V) |
| `train_mappo_v2.py` | MAPPO with the corrected update (separate optimizers, value normalization, 8 episodes per update) |
| `compare_algos.py` | Q-Learning, SARSA, DQN, PPO and action-masked PPO, e.g. `python compare_algos.py --algo dqn --seed 42` |
| `evaluate_all.py` | Evaluates baselines and all trained policies on 300 held-out scenarios, writes `final_results.json` |
| `make_figures.py` | Learning-curve figures for the report |
| `fill_report.py` | Fills every number in `report/report_template.tex` from `final_results.json` |
| `run_all.py` | Runs all of the above in order |
| `outputs/` | Trained runs (training CSV, model, evaluation) for every method and seed, plus `final_results.json` |
| `report/` | `final_report.tex` (paste into Overleaf), its template, and the figures |

## Using the report in Overleaf

Upload `report/final_report.tex`, `report/fig_comparison.pdf` and
`report/fig_learning_curves.pdf`. The architecture and flow diagrams are
referenced as `architecture.png` and `flowchart.png`; upload your existing
diagram images under those names.

## Notes

- Evaluation scenarios use seeds 900000-900299, disjoint from all training scenarios.
- Results vary slightly between machines and PyTorch versions; the conclusions should not.
- `fill_report.py` prints a WARNING if regenerated numbers no longer match a statement in the report text.
  The quick test always prints such warnings: 300 episodes is too short for the agents to learn, so it only checks that everything runs.
