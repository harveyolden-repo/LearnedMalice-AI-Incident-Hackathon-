# Simulated GRPO run

A mathematical simulation of a GRPO run over a training set containing
exploitable items.

- `alignment_sim.py` — the simulation: score distributions, question
  categories, and policy update rules.
- `sim_gui.py` — an interface for configuring a run and inspecting the
  resulting policy.

In the code the legitimate strategy is `l` and the exploit-based strategy is
`x`. `p_x` and `p_l` are the probabilities of opening with each; `p_cont_dict`
holds the conditional probabilities, keyed by `(first action, score bin)`.

## Requirements

Python 3.9 or later, `numpy`, and `tkinter`. Tkinter ships with Python on
Windows and macOS; on Debian or Ubuntu install it with
`sudo apt install python3-tk`.

```bash
pip install numpy
```

## Running

Keep both files in the same folder and run:

```bash
python sim_gui.py
```

The window takes a few seconds to appear, because `alignment_sim.py` executes
its own training loop at import time. The interface suppresses that output and
resets all state before each run. Indenting that loop under
`if __name__ == "__main__":` removes the delay.

## Controls

**Question mix.** Relative frequency of the four question categories,
normalised automatically, with the resulting percentage shown to the right.

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
log-scaled between 1e-5 and 0.999.

**Learning rates.** `lr_action` scales updates to the probability of the initial
action, `lr_conditional` scales updates to the conditional probability of the
second attempt, and `lr_boost` scales the legitimate and exploit strategy
advantages.

Every slider has a text field beside it — drag the slider, or type a value and
press Return. **Run** starts, **Stop** interrupts, **Defaults** restores every
field.

## Output

**P(x)** and **P(l)** are the probabilities of opening with an exploit or a
legitimate strategy. **boost_x** and **boost_l** are the accumulated exploit and
legitimate strategy advantages.

The table gives the probability of attempting an exploit on the second attempt:
the `after x` column is `P(X | X, N)`, the `after l` column is `P(X | L, N)`,
with the score `N` down the rows. Values update live during a run.

If a step contains no exploit attempt, no update is applied, so cells for
rarely visited states may remain at their initial value.
