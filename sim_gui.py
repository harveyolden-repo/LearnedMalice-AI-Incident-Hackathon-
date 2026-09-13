import contextlib
import io
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import alignment_sim as sim


QUESTION_TYPES = list(sim.problem_instances.keys())
ACTIONS = ("x", "l")
SCORE_BINS = tuple(range(1, 6))

# Defaults. Keyed by name; any class not listed falls back to the positional
# value in alignment_sim.question_probabilities.
_NAMED_DEFAULT_WEIGHTS = {"hard": 0.15, "l_only": 0.80, "both": 0.04,
                          "x_only": 0.01}
_POSITIONAL = dict(zip(QUESTION_TYPES, list(sim.question_probabilities)))
DEFAULT_WEIGHTS = {
    name: _NAMED_DEFAULT_WEIGHTS.get(name, _POSITIONAL.get(name, 0.0))
    for name in QUESTION_TYPES
}

DEFAULT_N_RUNS = "10000"
DEFAULT_BATCH = "16"
DEFAULT_P_X0 = 0.001
DEFAULT_LR_ACTION = "0.002"
DEFAULT_LR_CONDITIONAL = "0.002"
DEFAULT_LR_BOOST = "0.0001"

P_X0_MIN = 0.00001
P_X0_MAX = 0.99999


def fmt_num(value):
    """Compact but lossless-looking display for both 0.8 and 0.00001."""
    return f"{float(value):.6g}"


# ----------------------------------------------------------------------
# Simulation core (drives the imported module's globals)
# ----------------------------------------------------------------------

def configure(p_x0, lr_action, lr_conditional, lr_boost):
    """Reset the imported module's state before a fresh run."""
    sim.p_x = float(p_x0)
    sim.p_l = 1.0 - float(p_x0)

    sim.lr_action = float(lr_action)
    sim.lr_conditional = float(lr_conditional)
    sim.lr_boost = float(lr_boost)

    sim.performance_boost = {"x": 0.0, "l": 0.0}

    sim.p_cont_dict.clear()
    sim.p_cont_dict.update(
        {(a, s): float(p_x0) for a in ACTIONS for s in SCORE_BINS}
    )


def run_batch(question_type, batch_size):
    """One batch, mirroring the loop body in alignment_sim.py."""
    problem = sim.problem_instances[question_type]
    runs = []

    for _ in range(batch_size):
        action = sim.get_action()
        score1 = problem()[f"{action}_score"] + sim.performance_boost[action]
        score1 = float(np.clip(score1, 0, 5))

        act2 = sim.get_continued_action(action, score1)
        score2 = problem()[f"{act2}_score"] + sim.performance_boost[act2]
        score2 = float(np.clip(score2, 0, 5))

        runs.append({
            "act1": action,
            "score1": score1,
            "act2": act2,
            "score2": score2,
            "final_score": max(score1, score2),
        })

    batch_average = float(np.mean([r["final_score"] for r in runs]))
    for r in runs:
        r["batch_average"] = batch_average

    return runs


def snapshot(iteration, n_iterations):
    return {
        "iteration": iteration,
        "n_iterations": n_iterations,
        "p_x": float(sim.p_x),
        "p_l": float(sim.p_l),
        "boost": dict(sim.performance_boost),
        "p_cont": dict(sim.p_cont_dict),
    }


def worker(n_iterations, batch_size, weights, out_q, stop_event):
    """Runs in a background thread so the window stays responsive."""
    probs = np.asarray(weights, dtype=float)
    probs = probs / probs.sum()

    report_every = max(1, n_iterations // 200)

    try:
        for i in range(n_iterations):
            if stop_event.is_set():
                break

            question_type = sim.rng.choice(QUESTION_TYPES, p=probs)
            runs = run_batch(question_type, batch_size)

            sim.update_action_probabilities(runs)
            sim.update_conditional_probabilities(runs)
            sim.update_performance_boost(runs)

            if (i + 1) % report_every == 0:
                out_q.put(("progress", snapshot(i + 1, n_iterations)))

        out_q.put(("done", snapshot(i + 1, n_iterations)))
    except Exception as exc:  # surface errors in the status bar
        out_q.put(("error", repr(exc)))


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------

BG_LOW = (238, 242, 247)
BG_HIGH = (28, 110, 180)


def heat_colour(p):
    p = max(0.0, min(1.0, float(p)))
    rgb = [round(lo + (hi - lo) * p) for lo, hi in zip(BG_LOW, BG_HIGH)]
    fg = "white" if p > 0.55 else "#1b2733"
    return "#{:02x}{:02x}{:02x}".format(*rgb), fg


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None

        self.weight_vars = {}
        self.weight_entries = {}
        self.share_labels = {}
        self.cont_cells = {}

        self._build_controls()
        self._build_results()
        self._update_shares()
        self.after(100, self._poll)

    # -- controls ------------------------------------------------------

    def _build_controls(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))

        mix = ttk.LabelFrame(left, text="Question mix (normalised)", padding=8)
        mix.grid(row=0, column=0, sticky="ew")
        for r, name in enumerate(QUESTION_TYPES):
            self._weight_row(mix, r, name)

        run = ttk.LabelFrame(left, text="Run", padding=8)
        run.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.n_runs_var = tk.StringVar(value=DEFAULT_N_RUNS)
        self.batch_var = tk.StringVar(value=DEFAULT_BATCH)
        self.p_x0_var = tk.DoubleVar(value=DEFAULT_P_X0)
        self.p_x0_log = tk.DoubleVar(value=math.log10(DEFAULT_P_X0))

        self._entry_row(run, 0, "n_runs (iterations)", self.n_runs_var)
        self._entry_row(run, 1, "batch size", self.batch_var)

        ttk.Label(run, text="initial P(x)").grid(row=2, column=0, sticky="w")
        self.p_x0_scale = ttk.Scale(
            run, from_=math.log10(P_X0_MIN), to=math.log10(P_X0_MAX),
            variable=self.p_x0_log, length=120,
            command=lambda _e: self._p_x0_from_scale(),
        )
        self.p_x0_scale.grid(row=2, column=1, padx=6, pady=2, sticky="ew")
        self.p_x0_entry = ttk.Entry(run, width=10)
        self.p_x0_entry.grid(row=2, column=2, sticky="w")
        self.p_x0_entry.insert(0, fmt_num(DEFAULT_P_X0))
        self.p_x0_entry.bind("<Return>", lambda _e: self._p_x0_from_entry())
        self.p_x0_entry.bind("<FocusOut>", lambda _e: self._p_x0_from_entry())
        ttk.Label(run, text=f"slider is log10, {fmt_num(P_X0_MIN)}-"
                            f"{fmt_num(P_X0_MAX)}; type an exact value").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

        lrs = ttk.LabelFrame(left, text="Learning rates", padding=8)
        lrs.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.lr_action_var = tk.StringVar(value=DEFAULT_LR_ACTION)
        self.lr_cond_var = tk.StringVar(value=DEFAULT_LR_CONDITIONAL)
        self.lr_boost_var = tk.StringVar(value=DEFAULT_LR_BOOST)

        self._entry_row(lrs, 0, "lr_action", self.lr_action_var)
        self._entry_row(lrs, 1, "lr_conditional", self.lr_cond_var)
        self._entry_row(lrs, 2, "lr_boost", self.lr_boost_var)

        buttons = ttk.Frame(left)
        buttons.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.run_btn = ttk.Button(buttons, text="Run", command=self._start)
        self.run_btn.grid(row=0, column=0, sticky="w")
        self.stop_btn = ttk.Button(
            buttons, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)
        ttk.Button(
            buttons, text="Defaults", command=self._defaults
        ).grid(row=0, column=2)

    def _entry_row(self, parent, row, label, var):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=10).grid(
            row=row, column=1, columnspan=2, sticky="w", padx=6, pady=2)

    def _weight_row(self, parent, row, name):
        var = tk.DoubleVar(value=DEFAULT_WEIGHTS[name])
        self.weight_vars[name] = var

        ttk.Label(parent, text=name, width=8).grid(row=row, column=0, sticky="w")

        scale = ttk.Scale(
            parent, from_=0.0, to=1.0, variable=var, length=130,
            command=lambda _e, n=name: self._weight_from_scale(n),
        )
        scale.grid(row=row, column=1, padx=6, pady=2)

        entry = ttk.Entry(parent, width=8)
        entry.grid(row=row, column=2, padx=(0, 6))
        entry.insert(0, fmt_num(var.get()))
        entry.bind("<Return>", lambda _e, n=name: self._weight_from_entry(n))
        entry.bind("<FocusOut>", lambda _e, n=name: self._weight_from_entry(n))
        self.weight_entries[name] = entry

        lbl = ttk.Label(parent, text="", width=7)
        lbl.grid(row=row, column=3, sticky="w")
        self.share_labels[name] = lbl

    def _set_entry(self, entry, value):
        entry.delete(0, tk.END)
        entry.insert(0, fmt_num(value))

    def _weight_from_scale(self, name):
        self._set_entry(self.weight_entries[name], self.weight_vars[name].get())
        self._update_shares()

    def _weight_from_entry(self, name):
        var = self.weight_vars[name]
        try:
            value = min(1.0, max(0.0, float(self.weight_entries[name].get())))
        except ValueError:
            value = var.get()
        var.set(value)
        self._set_entry(self.weight_entries[name], value)
        self._update_shares()

    def _p_x0_from_scale(self):
        value = min(P_X0_MAX, max(P_X0_MIN, 10.0 ** self.p_x0_log.get()))
        self.p_x0_var.set(value)
        self._set_entry(self.p_x0_entry, value)

    def _p_x0_from_entry(self):
        try:
            value = min(P_X0_MAX, max(P_X0_MIN, float(self.p_x0_entry.get())))
        except ValueError:
            value = self.p_x0_var.get()
        self.p_x0_var.set(value)
        self.p_x0_log.set(math.log10(value))
        self._set_entry(self.p_x0_entry, value)

    def _commit_entries(self):
        """Pull any typed-but-uncommitted text into the variables."""
        for name in QUESTION_TYPES:
            self._weight_from_entry(name)
        self._p_x0_from_entry()

    def _update_shares(self):
        total = sum(v.get() for v in self.weight_vars.values())
        for name, var in self.weight_vars.items():
            if total <= 0:
                self.share_labels[name].config(text="--")
            else:
                self.share_labels[name].config(
                    text=f"{var.get() / total:.1%}")

    def _defaults(self):
        for name, var in self.weight_vars.items():
            var.set(DEFAULT_WEIGHTS[name])
            self._set_entry(self.weight_entries[name], DEFAULT_WEIGHTS[name])
        self.n_runs_var.set(DEFAULT_N_RUNS)
        self.batch_var.set(DEFAULT_BATCH)
        self.p_x0_var.set(DEFAULT_P_X0)
        self.p_x0_log.set(math.log10(DEFAULT_P_X0))
        self._set_entry(self.p_x0_entry, DEFAULT_P_X0)
        self.lr_action_var.set(DEFAULT_LR_ACTION)
        self.lr_cond_var.set(DEFAULT_LR_CONDITIONAL)
        self.lr_boost_var.set(DEFAULT_LR_BOOST)
        self._update_shares()

    # -- results -------------------------------------------------------

    def _build_results(self):
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        summary = ttk.LabelFrame(right, text="Result", padding=8)
        summary.grid(row=0, column=0, sticky="ew")
        summary.columnconfigure(1, weight=1)

        self.p_x_var = tk.StringVar(value="P(x) = --")
        ttk.Label(
            summary, textvariable=self.p_x_var,
            font=("TkDefaultFont", 14, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.p_x_bar = ttk.Progressbar(summary, maximum=1.0, length=260)
        self.p_x_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)

        self.boost_var = tk.StringVar(value="boost_x = --    boost_l = --")
        ttk.Label(summary, textvariable=self.boost_var).grid(
            row=2, column=0, columnspan=2, sticky="w")

        table = ttk.LabelFrame(
            right, text="P(next action = x | first action, first score)",
            padding=8)
        table.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        ttk.Label(table, text="score", width=7).grid(row=0, column=0)
        ttk.Label(table, text="after x", width=12, anchor="center").grid(
            row=0, column=1, padx=2)
        ttk.Label(table, text="after l", width=12, anchor="center").grid(
            row=0, column=2, padx=2)

        for r, score in enumerate(SCORE_BINS, start=1):
            ttk.Label(table, text=str(score), width=7, anchor="center").grid(
                row=r, column=0)
            for c, action in enumerate(ACTIONS, start=1):
                cell = tk.Label(
                    table, text="--", width=12, relief="flat",
                    bg="#eef2f7", fg="#1b2733", padx=4, pady=4)
                cell.grid(row=r, column=c, padx=2, pady=1, sticky="ew")
                self.cont_cells[(action, score)] = cell

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(right, textvariable=self.status_var).grid(
            row=2, column=0, sticky="w", pady=(10, 0))
        self.progress = ttk.Progressbar(right, maximum=1.0, length=260)
        self.progress.grid(row=3, column=0, sticky="ew", pady=(4, 0))

    def _show(self, snap):
        self.p_x_var.set(f"P(x) = {snap['p_x']:.4f}    P(l) = {snap['p_l']:.4f}")
        self.p_x_bar["value"] = snap["p_x"]
        self.boost_var.set(
            f"boost_x = {snap['boost']['x']:.4f}    "
            f"boost_l = {snap['boost']['l']:.4f}")

        for key, cell in self.cont_cells.items():
            p = snap["p_cont"].get(key)
            if p is None:
                continue
            bg, fg = heat_colour(p)
            cell.config(text=f"{p:.3f}", bg=bg, fg=fg)

        self.progress["value"] = snap["iteration"] / max(1, snap["n_iterations"])

    # -- run control ---------------------------------------------------

    def _read_params(self):
        self._commit_entries()
        n_runs = int(float(self.n_runs_var.get()))
        batch = int(float(self.batch_var.get()))
        if n_runs < 1 or batch < 1:
            raise ValueError("n_runs and batch size must be at least 1")

        weights = [self.weight_vars[n].get() for n in QUESTION_TYPES]
        if sum(weights) <= 0:
            raise ValueError("at least one question weight must be > 0")

        return {
            "n_runs": n_runs,
            "batch": batch,
            "weights": weights,
            "p_x0": float(self.p_x0_var.get()),
            "lr_action": float(self.lr_action_var.get()),
            "lr_cond": float(self.lr_cond_var.get()),
            "lr_boost": float(self.lr_boost_var.get()),
        }

    def _start(self):
        try:
            p = self._read_params()
        except ValueError as exc:
            self.status_var.set(f"Bad input: {exc}")
            return

        configure(p["p_x0"], p["lr_action"], p["lr_cond"], p["lr_boost"])

        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=worker,
            args=(p["n_runs"], p["batch"], p["weights"],
                  self.queue, self.stop_event),
            daemon=True,
        )
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set(f"Running {p['n_runs']} iterations...")
        self.thread.start()

    def _stop(self):
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def _finish(self, message):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set(message)

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._show(payload)
                    self.status_var.set(
                        f"Iteration {payload['iteration']} / "
                        f"{payload['n_iterations']}")
                elif kind == "done":
                    self._show(payload)
                    self._finish(
                        f"Finished at iteration {payload['iteration']}.")
                elif kind == "error":
                    self._finish(f"Error: {payload}")
        except queue.Empty:
            pass
        self.after(100, self._poll)


def main():
    root = tk.Tk()
    root.title("alignment_sim control panel")
    try:
        root.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
