# Simulated GRPO run

A mathematical simulation of a GRPO run over a training set containing
exploitable items. `sim_gui.py` provides an interface for configuring the run
and inspecting the resulting policy.

In the code, the legitimate strategy is `l` and the exploit-based strategy is
`x`. `p_x` and `p_l` are the probabilities of opening with each; `p_cont_dict`
holds the conditional probabilities, keyed by `(first action, score bin)`.

## Requirements

- Python 3.9 or later
- `numpy`
- `tkinter` — bundled with Python on Windows and macOS. On Debian or Ubuntu:
  ```bash
  sudo apt install python3-tk
  ```
- `matplotlib` — needed only by `plot_trajectory.py`, not by the interface.

```bash
pip install numpy matplotlib
```

## Running the interface

Keep `sim_gui.py` in the same folder as `alignment_sim.py` and run:

```bash
python sim_gui.py
```

The window takes a few seconds to appear, because `alignment_sim.py` executes
its own training loop at import time. The interface suppresses that output and
resets all state before each run. Indenting that loop under
`if __name__ == "__main__":` removes the delay.

## Controls

**Question mix.** Relative frequency of the four question categories, which are
normalised automatically; the percentage each contributes is shown to the right.
Every slider has a text field beside it — drag the slider, or type a value and
press Return.

| slider | category |
|---|---|
| `hard` | Correct solution difficult, exploit difficult |
| `l_only` | Legitimate Only: correct solution medium, exploit hard |
| `both` | Correct solution medium, exploit medium |
| `x_only` | Exploit Only: correct solution difficult, exploit medium |

**Run.** `n_runs` is the number of training steps. `batch size` is the number of
model responses generated per step, over which the relative advantage is
computed.

**initial P(x).** The probability of opening with an exploit before training.
Every conditional probability is initialised to this value. The slider is
log-scaled between 1e-5 and 0.999; type into the box for an exact value.

**Learning rates.** `lr_action` scales updates to the probability of the initial
action, `lr_conditional` scales updates to the conditional probability of the
second attempt, and `lr_boost` scales the legitimate and exploit strategy
advantages.

Press **Run** to start, **Stop** to interrupt, and **Defaults** to restore every
field.

## Reading the output

**P(x)** and **P(l)** are the probabilities of opening with an exploit or a
legitimate strategy. **boost_x** and **boost_l** are the accumulated exploit and
legitimate strategy advantages.

The table gives the probability of attempting an exploit on the second attempt,
for each combination of first attempt and observed score — the `after x` column
is `P(X | X, N)` and the `after l` column is `P(X | L, N)`, with `N` down the
rows. Cells are shaded by value. Results update live during a run, and the
status line reports the step reached.

Note that if a step contains no exploit attempt, no update is applied, so cells
representing rarely visited states may remain at their initial value.

## Reproducing the report figures

Run each script from the folder containing `alignment_sim.py`. Both analysis
scripts call the module's own update functions rather than reimplementing them.

| script | output |
|---|---|
| `sweep_p0.py` | Sweeps initial `P(x)` over six values with five seeds each; writes `sweep_results.json`. Takes roughly three minutes. |
| `make_sweep_table.py` | Renders `sweep_results.json` as `fig_sweep_p0.pdf`. |
| `plot_trajectory.py` | Traces `P(X \| L, 1)` across training steps for several seeds; writes `fig_trajectory.pdf`. |
| `make_figures.py` | Writes `fig_parameters.pdf` and `fig_pipeline.pdf`. |

Settings for each are defined as constants at the top of the corresponding
script. `sweep_p0.py` reads the question mix and batch size from
`alignment_sim.py`, so edit them there.

## Repository contents

| file | purpose |
|---|---|
| `alignment_sim.py` | The simulation: score distributions, question categories, and policy update rules. |
| `sim_gui.py` | The interface. |
| `sweep_p0.py`, `make_sweep_table.py` | Initial `P(x)` sweep and its table. |
| `plot_trajectory.py` | Convergence figure. |
| `make_figures.py` | Parameter table and simulation structure figure. |
| `sweep_results.json` | Raw output of the sweep reported in the paper. |
