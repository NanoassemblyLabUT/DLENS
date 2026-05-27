"""Probability distribution popup window."""

from __future__ import annotations

import tkinter as tk
from tkinter import Entry, Label, Toplevel

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ProbabilityWindowMixin:
    """Creates the optional probability distribution sub-window."""

    _probability_attrs = (
        "entry_0_m", "entry_0_s", "entry_0_d",
        "entry_1_m", "entry_1_s", "entry_1_d",
        "entry_2_m", "entry_2_s", "entry_2_d",
        "label_1_m", "label_1_s",
        "figure_0", "figure_1", "figure_2",
        "plot_0", "plot_1", "plot_2",
        "canvas_0", "canvas_1", "canvas_2",
    )

    def _HasProbabilityWindow(self) -> bool:
        window = getattr(self, "probability_window", None)
        return window is not None and bool(window.winfo_exists())

    def _CloseProbabilityWindow(self) -> None:
        window = getattr(self, "probability_window", None)
        if window is not None and window.winfo_exists():
            window.destroy()
        self.probability_window = None
        for attr in self._probability_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

    def _OpenProbabilityWindow(self, *args, **kwargs) -> None:
        if self._HasProbabilityWindow():
            self.probability_window.lift()
            return None

        window = Toplevel(self.parent)
        window.title("Parameter Probabilities")
        window.geometry("1260x380")
        window.protocol("WM_DELETE_WINDOW", self._CloseProbabilityWindow)
        self.probability_window = window

        self._LayProbabilityWindow(window)
        self._Reconfigure()

        if getattr(self, "fitted", False):
            self._DisplayProbability()
            self._Draw_probability()

        return None

    def _LayProbabilityWindow(self, parent: Toplevel) -> None:
        reg = parent.register(self._Callback)
        labels = (
            ("Radius Probability Distribution", "Radius (A)", "Mean (A):", "STD (A):"),
            ("Shape Probability Distribution", "Shape", "Mean:", "STD:"),
            ("PDI Probability Distribution", "PDI", "Mean:", "STD:"),
        )
        x_positions = (16, 432, 848)

        entries = []
        figures = []
        plots = []
        canvases = []
        mean_labels = []
        std_labels = []

        for idx, (x, label_info) in enumerate(zip(x_positions, labels)):
            title, xlabel, mean_label, std_label = label_info
            figure = Figure(figsize=(4, 3), dpi=64)
            plot = figure.add_subplot(1, 1, 1)
            plot.set_title(title)
            plot.set_xlabel(xlabel)
            plot.set_ylabel("Probability Density")
            if idx == 2:
                plot.set_xscale("log")

            canvas = FigureCanvasTkAgg(figure, parent)
            canvas.get_tk_widget().place(height=280, width=400, x=x, y=0)

            label_m = Label(parent, text=mean_label)
            label_m.place(height=30, width=64, x=x, y=292)
            entry_m = Entry(parent)
            entry_m.place(height=30, width=64, x=x + 64, y=292)
            entry_m.config(state=tk.DISABLED, bg="Light grey", validate="key", validatecommand=(reg, "%P"))

            label_s = Label(parent, text=std_label)
            label_s.place(height=30, width=64, x=x + 128, y=292)
            entry_s = Entry(parent)
            entry_s.place(height=30, width=64, x=x + 192, y=292)
            entry_s.config(state=tk.DISABLED, bg="Light grey", validate="key", validatecommand=(reg, "%P"))

            label_d = Label(parent, text="Deviation:")
            label_d.place(height=30, width=64, x=x + 256, y=292)
            entry_d = Entry(parent)
            entry_d.place(height=30, width=64, x=x + 320, y=292)
            entry_d.config(state=tk.DISABLED, bg="Light grey", validate="key", validatecommand=(reg, "%P"))

            entries.append((entry_m, entry_s, entry_d))
            figures.append(figure)
            plots.append(plot)
            canvases.append(canvas)
            mean_labels.append(label_m)
            std_labels.append(label_s)

        self.entry_0_m, self.entry_0_s, self.entry_0_d = entries[0]
        self.entry_1_m, self.entry_1_s, self.entry_1_d = entries[1]
        self.entry_2_m, self.entry_2_s, self.entry_2_d = entries[2]

        self.label_1_m = mean_labels[1]
        self.label_1_s = std_labels[1]

        self.figure_0, self.figure_1, self.figure_2 = figures
        self.plot_0, self.plot_1, self.plot_2 = plots
        self.canvas_0, self.canvas_1, self.canvas_2 = canvases

        return None
