"""Subclass probability popup window."""

from __future__ import annotations

import tkinter as tk
from tkinter import Entry, Frame, Label, Toplevel


class SubclassProbabilityWindowMixin:
    """Creates the optional subclass probability sub-window."""

    def _HasSubclassProbabilityWindow(self) -> bool:
        window = getattr(self, "subclass_probability_window", None)
        return window is not None and bool(window.winfo_exists())

    def _CloseSubclassProbabilityWindow(self) -> None:
        window = getattr(self, "subclass_probability_window", None)
        if window is not None and window.winfo_exists():
            window.destroy()
        self.subclass_probability_window = None
        self.subclass_probability_rows = []

    def _OpenSubclassProbabilityWindow(self, *args, **kwargs) -> None:
        if self._HasSubclassProbabilityWindow():
            self.subclass_probability_window.lift()
            self._DisplaySubclassProbabilities()
            return None

        window = Toplevel(self.parent)
        window.title("Subclass Probabilities")
        window.geometry("420x220")
        window.protocol("WM_DELETE_WINDOW", self._CloseSubclassProbabilityWindow)
        self.subclass_probability_window = window

        self._LaySubclassProbabilityWindow(window)
        self._DisplaySubclassProbabilities()
        return None

    def _LaySubclassProbabilityWindow(self, parent: Toplevel) -> None:
        title = Label(parent, text="Subclass probabilities", anchor="w")
        title.place(height=30, width=380, x=16, y=8)
        self.subclass_probability_title = title
        self.subclass_probability_rows = []
        return None

    def _DisplaySubclassProbabilities(self, *args, **kwargs) -> None:
        if not self._HasSubclassProbabilityWindow():
            return None

        parent = self.subclass_probability_window
        for row in getattr(self, "subclass_probability_rows", []):
            for widget in row:
                widget.destroy()
        self.subclass_probability_rows = []

        rows = self._SubclassProbabilityRows()
        if not rows:
            label = Label(parent, text="No subclass probabilities are available.", anchor="w")
            label.place(height=30, width=360, x=16, y=52)
            self.subclass_probability_rows.append((label,))
            return None

        for index, (name, probability) in enumerate(rows):
            y = 48 + 34 * index
            label = Label(parent, text=f"{name}:", anchor="w")
            label.place(height=26, width=176, x=16, y=y)

            entry = Entry(parent)
            entry.place(height=26, width=72, x=196, y=y)
            entry.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            entry.delete(0, tk.END)
            entry.insert(0, "N/A" if probability is None else f"{100 * probability:.3f}%")
            entry.config(state=tk.DISABLED, disabledbackground="Light grey", disabledforeground="black")

            bar_outer = Frame(parent, bd=1, relief=tk.SUNKEN, background="white")
            bar_outer.place(height=18, width=112, x=284, y=y + 4)
            bar_width = 0 if probability is None else max(0, min(108, int(108 * probability)))
            bar_inner = Frame(parent, background="#5aa469")
            bar_inner.place(height=14, width=bar_width, x=286, y=y + 6)

            self.subclass_probability_rows.append((label, entry, bar_outer, bar_inner))

        return None