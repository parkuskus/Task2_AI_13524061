"""GUI Interaktif Visualisasi Local Search (Bonus #2 Spec).

Fitur:
  1. Grafik konvergensi J(s) vs iterasi
  2. Visualisasi state (r_t vs quarter) animasi real-time
  3. Panel status constraint (hijau = PASS, merah = FAIL)
  4. Kontrol hyperparameter interaktif (slider + nilai real-time)

Usage:
    python gui_app.py
"""

import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import tkinter as tk
from tkinter import ttk

from config import DEFAULT_PARAMS
from economy.shocks import generate_shocks
from economy.simulation import simulate_economy
from evaluation.constraints import check_constraints
from search.utils import generate_initial_state
from search.hill_climbing import hill_climbing
from search.simulated_annealing import simulated_annealing
from search.genetic_algorithm import genetic_algorithm

HC_VARIANTS = ["steepest_ascent", "sideways", "stochastic", "random_restart"]
ALGOS = ["Hill-Climbing", "Simulated Annealing", "Genetic Algorithm"]


class LocalSearchGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Local Search Visualizer | KBI Interest Rate Optimisation")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.params = dict(DEFAULT_PARAMS)
        self.T = self.params["T"]
        self.shocks = generate_shocks(T=self.T, seed=42)
        self.quarters = list(range(1, self.T + 1))

        self.running = False
        self._stop_flag = False
        self._current_state = None
        self._current_score = float("-inf")
        self._iter_count = 0
        self._stored_scores = []
        self._algo_key = "HC"

        self._build_ui()
        self.root.mainloop()

    # ---- UI Construction ----

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1, minsize=260)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        ctrl_frame = ttk.LabelFrame(self.root, text="Controls", padding=8)
        ctrl_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._build_algo_selector(ctrl_frame)
        self._build_hyperparams(ctrl_frame)
        self._build_viz_controls(ctrl_frame)
        self._build_buttons(ctrl_frame)
        self._build_constraint_panel(ctrl_frame)

        self.fig, (self.ax_conv, self.ax_state) = plt.subplots(2, 1, figsize=(9, 7.5))
        self.fig.subplots_adjust(hspace=0.40, top=0.93, bottom=0.08, left=0.10, right=0.97)
        self.fig.suptitle("Local Search Visualization", fontsize=12, fontweight="bold")

        self._setup_conv_plot()
        self._setup_state_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

    def _build_algo_selector(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 8))

        ttk.Label(frame, text="Algorithm:").pack(anchor="w")
        self.algo_var = tk.StringVar(value=ALGOS[0])
        self.algo_combo = ttk.Combobox(frame, textvariable=self.algo_var,
                                       values=ALGOS, state="readonly")
        self.algo_combo.pack(fill="x", pady=(2, 4))
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_change)

        ttk.Label(frame, text="HC Variant:").pack(anchor="w")
        self.hc_var = tk.StringVar(value=HC_VARIANTS[0])
        self.hc_combo = ttk.Combobox(frame, textvariable=self.hc_var,
                                     values=HC_VARIANTS, state="readonly")
        self.hc_combo.pack(fill="x")

    def _build_hyperparams(self, parent):
        self.hp_frame = ttk.LabelFrame(parent, text="Hyperparameters", padding=6)
        self.hp_frame.pack(fill="x", pady=(0, 8))

        self._hp_entries = {}
        self._add_hp_slider("max_iter", "Max Iterations", 50, 5000, 1000, is_int=True)

        self.algo_hp = ttk.Frame(self.hp_frame)
        self.algo_hp.pack(fill="x", pady=(0, 4))

        self.hc_frame = ttk.Frame(self.algo_hp)
        self._add_hp_slider("hc_patience", "HC: Patience", 50, 2000, 500, is_int=True, parent=self.hc_frame)

        self.sa_frame = ttk.Frame(self.algo_hp)
        self._add_hp_slider("sa_T0", "SA: Initial Temp (T0)", 1.0, 200.0, 50.0, parent=self.sa_frame)
        self._add_hp_slider("sa_cool", "SA: Cooling Rate", 0.900, 0.9995, 0.998, parent=self.sa_frame)

        self.ga_frame = ttk.Frame(self.algo_hp)
        self._add_hp_slider("ga_pop", "GA: Population Size", 10, 300, 100, is_int=True, parent=self.ga_frame)
        self._add_hp_slider("ga_mut", "GA: Mutation Rate", 0.05, 0.60, 0.25, parent=self.ga_frame)

        self._add_hp_slider("w_pi", "Weight: Inflation (w_pi)", 0.1, 5.0, 1.0)
        self._add_hp_slider("w_y", "Weight: Output Gap (w_y)", 0.1, 5.0, 0.5)
        self._add_hp_slider("w_pp", "Weight: Purchasing Power (w_pp)", 0.1, 5.0, 0.3)
        self._add_hp_slider("w_r", "Weight: Smoothing (w_r)", 0.1, 5.0, 0.2)

        for f in [self.hc_frame, self.sa_frame, self.ga_frame]:
            for w in f.winfo_children():
                w.pack(fill="x", pady=1)

        self._on_algo_change()

    def _build_viz_controls(self, parent):
        self.viz_frame = ttk.LabelFrame(parent, text="Visualization Speed", padding=6)
        self.viz_frame.pack(fill="x", pady=(0, 8))
        self._viz_entries = {}
        self._add_viz_slider("viz_speed", "GUI Refresh (ms/frame)", 50, 500, 150, is_int=True)
        self._add_viz_slider("step_delay", "Step Delay (ms/iter)", 0, 500, 80, is_int=True)

    def _add_hp_slider(self, key, label, lo, hi, default, is_int=False, parent=None):
        return self._add_slider_impl(key, label, lo, hi, default, is_int, parent, self._hp_entries)

    def _add_viz_slider(self, key, label, lo, hi, default, is_int=False):
        return self._add_slider_impl(key, label, lo, hi, default, is_int, self.viz_frame, self._viz_entries)

    def _add_slider_impl(self, key, label, lo, hi, default, is_int, parent, registry):
        if parent is None:
            parent = self.hp_frame
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        var = tk.DoubleVar(value=default)
        val_lbl = tk.StringVar(value=self._fmt_val(default, is_int))

        def on_move(*_):
            v = var.get()
            val_lbl.set(self._fmt_val(v, is_int))

        lbl = ttk.Label(row, text=label, font=("", 8))
        lbl.pack(anchor="w")
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                          command=on_move)
        scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
        val_display = ttk.Label(row, textvariable=val_lbl, font=("", 8, "bold"),
                                width=8, anchor="e")
        val_display.pack(side="right")

        entry = {"var": var, "is_int": is_int, "scale": scale, "label": lbl, "val": val_display}
        registry[key] = entry
        return entry

    @staticmethod
    def _fmt_val(v, is_int):
        return str(int(round(v))) if is_int else f"{v:.4f}"

    def _get_entry_value(self, entries, key):
        entry = entries[key]
        val = entry["var"].get()
        return int(round(val)) if entry["is_int"] else val

    def _get_hp_value(self, key):
        return self._get_entry_value(self._hp_entries, key)

    def _get_viz_value(self, key):
        return self._get_entry_value(self._viz_entries, key)

    def _set_hp_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for entry in self._hp_entries.values():
            try:
                entry["scale"].config(state=state)
                entry["label"].config(foreground="black" if enabled else "#a0a0a0")
            except tk.TclError:
                pass
        try:
            self.algo_combo.config(state="readonly" if enabled else "disabled")
            self.hc_combo.config(state="readonly" if (enabled and self.algo_var.get() == "Hill-Climbing") else "disabled")
        except tk.TclError:
            pass

    def _set_viz_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for entry in self._viz_entries.values():
            try:
                entry["scale"].config(state=state)
                entry["label"].config(foreground="black" if enabled else "#a0a0a0")
            except tk.TclError:
                pass

    def _set_all_controls_enabled(self, enabled):
        self._set_hp_enabled(enabled)
        self._set_viz_enabled(enabled)

    def _build_buttons(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_frame, text="RUN Search", command=self._start_search).pack(
            fill="x", pady=2)
        ttk.Button(btn_frame, text="STOP", command=self._stop_search).pack(
            fill="x", pady=2)

        self.progress_var = tk.StringVar(value="Ready.")
        ttk.Label(btn_frame, textvariable=self.progress_var, font=("", 8)).pack(anchor="w")

    def _build_constraint_panel(self, parent):
        self.cp_frame = ttk.LabelFrame(parent, text="Constraint Status (current state)", padding=6)
        self.cp_frame.pack(fill="x", pady=(0, 8))
        self.constraint_labels = {}
        labels = {
            "C1": "Batas r_t [3%,8%], 25 bps",
            "C2": "CA >= -3% PDB",
            "C3": "Spread SBN <= 2.5%",
            "C4": "|pi_T - 2.5%| <= 1%",
            "C5": "Smoothing <= 50 bps (SOFT)",
        }
        for cid, desc in labels.items():
            row = ttk.Frame(self.cp_frame)
            row.pack(fill="x", pady=1)
            lbl = ttk.Label(row, text=f" {cid}: {desc}", font=("", 8))
            lbl.pack(side="left")
            self.constraint_labels[cid] = lbl
        self._set_constraint_status("unknown")

    # ---- Plot initialisation ----

    def _setup_conv_plot(self):
        self.ax_conv.clear()
        self.ax_conv.set_title("Convergence: J(s) vs Iteration", fontsize=10)
        self.ax_conv.set_xlabel("Iteration")
        self.ax_conv.set_ylabel("J(s)")
        self.ax_conv.grid(alpha=0.3)
        self.ax_conv.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
        self._conv_lines = {}
        for algo, color in zip(["HC", "SA", "GA"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            line, = self.ax_conv.plot([], [], color=color, linewidth=1.0, label=algo, alpha=0.7)
            self._conv_lines[algo] = line
        self.ax_conv.legend(fontsize=7, loc="lower right")

    def _setup_state_plot(self):
        self.ax_state.clear()
        self.ax_state.set_title("State Trajectory: BI-Rate per Quarter", fontsize=10)
        self.ax_state.set_xlabel("Quarter")
        self.ax_state.set_ylabel("BI-Rate (%)")
        self.ax_state.set_xticks(self.quarters)
        self.ax_state.set_ylim(2.5, 8.5)
        self.ax_state.axhline(y=self.params["r0"], color="gray", linestyle=":",
                              alpha=0.5, label=f"r0={self.params['r0']}%")
        self.ax_state.axhline(y=self.params["pi_star"], color="green", linestyle="--",
                              alpha=0.3, label=f"pi*={self.params['pi_star']}%")
        self.ax_state.fill_between(self.quarters, self.params["r_min"],
                                   self.params["r_max"], alpha=0.05, color="green")
        self.ax_state.legend(fontsize=7, loc="lower left")
        self._state_line, = self.ax_state.plot([], [], "o-", color="#1f77b4", linewidth=2, markersize=6)

    # ---- Constraint status ----

    def _set_constraint_status(self, status):
        colors = {"pass": "#27ae60", "fail": "#e74c3c", "soft_ok": "#27ae60",
                  "soft_viol": "#e67e22", "unknown": "#bdc3c7"}
        if status == "unknown":
            for lbl in self.constraint_labels.values():
                lbl.config(foreground="#bdc3c7")
            return
        for cid, ok in status.items():
            if cid == "C5":
                color = colors["soft_ok"] if ok else colors["soft_viol"]
            else:
                color = colors["pass"] if ok else colors["fail"]
            if cid in self.constraint_labels:
                self.constraint_labels[cid].config(foreground=color)

    def _update_constraint_from_state(self, state):
        if state is None:
            self._set_constraint_status("unknown")
            return
        traj = simulate_economy(state, self.shocks, self.params)
        cons = check_constraints(state, traj, self.params)
        self._set_constraint_status({"C1": cons["C1"], "C2": cons["C2"],
                                     "C3": cons["C3"], "C4": cons["C4"], "C5": cons["C5"]})

    # ---- Algorithm dispatch ----

    def _on_algo_change(self, event=None):
        algo = self.algo_var.get()
        self.hc_combo.config(state="readonly" if algo == "Hill-Climbing" else "disabled")

        for f in [self.hc_frame, self.sa_frame, self.ga_frame]:
            for w in f.winfo_children():
                w.pack_forget()
            f.pack_forget()

        if algo == "Hill-Climbing":
            self.hc_frame.pack(fill="x")
            for w in self.hc_frame.winfo_children():
                w.pack(fill="x", pady=1)
        elif algo == "Simulated Annealing":
            self.sa_frame.pack(fill="x")
            for w in self.sa_frame.winfo_children():
                w.pack(fill="x", pady=1)
        else:
            self.ga_frame.pack(fill="x")
            for w in self.ga_frame.winfo_children():
                w.pack(fill="x", pady=1)

    def _build_params(self):
        p = dict(self.params)
        p["w_pi"] = self._get_hp_value("w_pi")
        p["w_y"] = self._get_hp_value("w_y")
        p["w_pp"] = self._get_hp_value("w_pp")
        p["w_r"] = self._get_hp_value("w_r")
        return p

    # ---- Search execution ----

    def _start_search(self):
        if self.running:
            return
        self._stop_flag = False
        self.running = True
        self._iter_count = 0
        self._current_state = None
        self._current_score = float("-inf")
        self._stored_scores = []

        self._set_all_controls_enabled(False)
        self._setup_conv_plot()
        self._setup_state_plot()
        self._last_drawn_iter = -1
        self.canvas.draw_idle()

        self._set_constraint_status("unknown")
        self.progress_var.set("Searching...")

        self._search_thread = threading.Thread(target=self._run_search, daemon=True)
        self._search_thread.start()
        self._poll_done = False
        self._poll_update()

    def _stop_search(self):
        self._stop_flag = True

    def _run_search(self):
        p = self._build_params()
        max_iter = self._get_hp_value("max_iter")
        rng = np.random.default_rng(42)
        s0 = generate_initial_state(T=self.T, rng=rng)
        algo = self.algo_var.get()
        self._algo_key = algo_key = {"Hill-Climbing": "HC", "Simulated Annealing": "SA",
                                      "Genetic Algorithm": "GA"}[algo]
        stored_scores = []
        self._stored_scores = stored_scores
        step_delay = self._get_viz_value("step_delay") / 1000.0

        def on_iter(state, score, it):
            if self._stop_flag:
                return
            self._current_state = list(state)
            self._current_score = score
            self._iter_count = it + 1
            stored_scores.append((it, score))
            if step_delay > 0:
                time.sleep(step_delay)

        try:
            if algo == "Hill-Climbing":
                hc_patience = min(self._get_hp_value("hc_patience"), max_iter)
                self._result = hill_climbing(
                    s0, max_iter=max_iter, shocks=self.shocks, params=p,
                    variant=self.hc_var.get(), patience=hc_patience,
                    on_iteration=on_iter,
                )
            elif algo == "Simulated Annealing":
                self._result = simulated_annealing(
                    s0, max_iter=max_iter,
                    T0=self._get_hp_value("sa_T0"),
                    cooling_rate=self._get_hp_value("sa_cool"),
                    shocks=self.shocks, params=p,
                    on_iteration=on_iter,
                )
            else:
                generations = max(1, max_iter)
                pop_size = self._get_hp_value("ga_pop")
                self._result = genetic_algorithm(
                    pop_size=pop_size, generations=generations,
                    mutation_rate=self._get_hp_value("ga_mut"),
                    shocks=self.shocks, params=p,
                    on_iteration=on_iter,
                )
        except Exception:
            self._result = None
        finally:
            self.root.after(0, self._search_done)

    def _search_done(self):
        self.running = False
        self._set_all_controls_enabled(True)
        self._poll_done = True
        if self._result is None:
            self.progress_var.set("Error during search.")
            return
        self.progress_var.set(
            f"Done. Best J(s)={self._result['best_score']:.2f} | "
            f"Iters={self._result['iterations']}"
        )
        self._update_constraint_from_state(self._result["best_state"])
        self.canvas.draw_idle()

    def _poll_update(self):
        if not self.running:
            if not self._poll_done:
                self._poll_done = True
                self._set_all_controls_enabled(True)
            return

        state = self._current_state
        if state is None:
            delay = max(30, self._get_viz_value("viz_speed"))
            self.root.after(delay, self._poll_update)
            return

        current_iter = self._iter_count
        if current_iter == self._last_drawn_iter:
            delay = max(30, self._get_viz_value("viz_speed"))
            self.root.after(delay, self._poll_update)
            return

        color = {"Hill-Climbing": "#1f77b4",
                 "Simulated Annealing": "#ff7f0e",
                 "Genetic Algorithm": "#2ca02c"}.get(self.algo_var.get(), "#1f77b4")

        # State trajectory (lightweight: just set_data)
        self._state_line.set_data(self.quarters, state)
        self._state_line.set_color(color)
        self.ax_state.set_title(f"State Trajectory (iter {current_iter})", fontsize=10)

        # Convergence curve (decimated)
        if self._stored_scores:
            xs, ys = zip(*self._stored_scores)
            n = len(xs)
            max_pts = 300
            if n > max_pts:
                idxs = sorted(set(int(round(i * (n - 1) / (max_pts - 1))) for i in range(max_pts)))
                xs = [xs[i] for i in idxs]
                ys = [ys[i] for i in idxs]
            self._conv_lines[self._algo_key].set_data(xs, ys)
            self.ax_conv.relim()
            self.ax_conv.autoscale_view()

        self._update_constraint_from_state(state)
        self.progress_var.set(f"Iter {current_iter} | J(s) = {self._current_score:.2f}")
        self._last_drawn_iter = current_iter

        self.canvas.draw_idle()
        delay = max(30, self._get_viz_value("viz_speed"))
        self.root.after(delay, self._poll_update)

    def _on_close(self):
        self._stop_flag = True
        self.running = False
        self.root.quit()
        self.root.destroy()
        os._exit(0)


if __name__ == "__main__":
    LocalSearchGUI()