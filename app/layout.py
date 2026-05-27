"""Main window layout mixin."""

from __future__ import annotations

import tkinter as tk
from tkinter import Button, Entry, Label, OptionMenu, StringVar

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.widgets import ToolTip
from inference.shape_registry import SHAPES


class LayoutMixin:
    """Methods that create the main SAXS analysis window layout."""

    def _LayOperation(self, *args, **kwargs) -> None:

        parent = self.parent
        dy = self.dy

        button_folder = Button(parent, text="Load Folder", command=self._LoadFolder)
        button_folder.place(height=2 * dy, width=128, x=16, y=1 * dy)
        ToolTip(button_folder, "Click to load a folder")

        button_file = Button(parent, text="Load File", command=self._LoadFile)
        button_file.place(height=2 * dy, width=128, x=16, y=3 * dy)
        ToolTip(button_file, "Click to load a file")

        shapes = [spec.display_name for spec in SHAPES.values() if spec.model_key is not None]
        select_shapes = StringVar()
        select_shapes.set("Subclass")

        select_family = StringVar()
        select_family.set("Main Class")

        self.old_mode = None

        drop_methods = OptionMenu(parent, select_shapes, *shapes, command=self._Drop_Fit)
        drop_methods.config(width=20)
        drop_methods.place(height=30, width=128, x=16, y=5 * dy)
        drop_methods.config(state=tk.DISABLED, bg="Light grey")

        button_clear = Button(parent, text="Clear", command=self._Clear)
        button_clear.place(height=30, width=128, x=16, y=7 * dy)
        button_clear.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_clear, "Click to clear the input fields")

        button_export = Button(parent, text="Export", command=self._Export)
        button_export.place(height=30, width=128, x=16, y=9 * dy)
        button_export.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_export, "Click to export the data")

        button_simulate = Button(parent, text="Start", command=self._Simulate_as)
        button_simulate.place(height=30, width=128, x=16, y=11 * dy)
        button_simulate.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_simulate, "Click to start the simulation")

        button_visualize = Button(parent, text="Visualize", command=self._Visualize)
        button_visualize.place(height=30, width=128, x=16, y=13 * dy)
        button_visualize.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_visualize, "Click to visualize the results")

        button_probability = Button(parent, text="Probabilities", command=self._OpenProbabilityWindow)
        button_probability.place(height=30, width=140, x=16, y=15 * dy) 
        button_probability.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_probability, "Click to open parameter probability plots")

        label_count = Label(parent, text="N/A")
        label_count.place(height=30, width=64, x=16, y=17 * dy)

        button_backward = Button(parent, text='<', command=self._Backward)
        button_backward.place(height=30, width=30, x=80, y=17 * dy)
        button_backward.config(state=tk.DISABLED, bg="Light grey")

        button_forward = Button(parent, text='>', command=self._Forward)
        button_forward.place(height=30, width=30, x=112, y=17 * dy)
        button_forward.config(state=tk.DISABLED, bg="Light grey")

        label_MSLE = Label(parent, text="mMSLE:")
        label_MSLE.place(height=30, width=64, x=144, y=17 * dy)
        ToolTip(label_MSLE, "Displays the Mean Squared Logarithmic Error")

        var_MSLE = StringVar()

        Entry_MSLE = Entry(parent, textvariable=var_MSLE)
        Entry_MSLE.place(height=30, width=96, x=208, y=17 * dy)
        Entry_MSLE.config(state=tk.DISABLED, disabledbackground="Light grey", disabledforeground="black")

        family_choices = ("Isotropic", "Anisotropic")
        drop_families = OptionMenu(parent, select_family, *family_choices, command=self._Drop_Family)
        drop_families.config(width=20)
        drop_families.place(height=30, width=140, x=320, y=17 * dy)
        drop_families.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(drop_families, "Select the main shape class")

        button_subclass_probability = Button(parent, text="Subclasses", command=self._OpenSubclassProbabilityWindow)
        button_subclass_probability.place(height=30, width=110, x=480, y=17 * dy)
        button_subclass_probability.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_subclass_probability, "Click to open subclass probabilities")

        label_comment = Label(parent, text="Comment:")
        label_comment.place(height=30, width=128, x=16, y=19 * dy)

        var_comment = StringVar()

        entry_comment = Entry(parent, textvariable=var_comment)
        entry_comment.place(height=30, width=192, x=144, y=19 * dy)
        entry_comment.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

        label_class_0 = Label(parent, text="Isotropic:")
        label_class_0.place(height=30, width=80, x=350, y=19 * dy)
        ToolTip(label_class_0, "Probability of being Isotropic")

        var__class0 = StringVar()

        entry_class_0 = Entry(parent, textvariable=var__class0)
        entry_class_0.place(height=30, width=64, x=430, y=19 * dy)
        entry_class_0.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(entry_class_0, "Probability of being Isotropic")

        label_class_1 = Label(parent, text="Anisotropic:")
        label_class_1.place(height=30, width=100, x=510, y=19 * dy)
        ToolTip(label_class_1, "Probability of being Anisotropic")

        var__class1 = StringVar()

        entry_class_1 = Entry(parent, textvariable=var__class1)
        entry_class_1.place(height=30, width=64, x=610, y=19 * dy)
        entry_class_1.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(entry_class_1, "Probability of being Anisotropic")

        button_autosub = Button(parent, text='Auto-subtraction', command=self._Autosubtraction)
        button_autosub.place(height=30, width=140, x=680, y=9 * dy)
        ToolTip(button_autosub, "Click to perform auto-subtraction")
        
        button_smooth = Button(parent, text='Smoothen', command=self._Smoothen)
        button_smooth.place(height=30, width=140, x=680, y=11 * dy)
        ToolTip(button_smooth, "Click to smoothen data")

        button_help = Button(parent, text='Help', command=self._Help)
        button_help.place(height=30, width=140, x=680, y=13 * dy)
        ToolTip(button_help, "Click to access help")

        self.drop_methods = drop_methods
        self.drop_families = drop_families
        
        self.select_shapes = select_shapes
        self.select_family = select_family

        self.button_file = button_file
        self.button_folder = button_folder
        self.button_clear = button_clear
        self.button_export = button_export
        self.button_simulate = button_simulate
        self.button_visualize = button_visualize
        self.button_probability = button_probability
        self.button_subclass_probability = button_subclass_probability
        self.button_forward = button_forward
        self.button_backward = button_backward

        self.Entry_MSLE = Entry_MSLE
        self.entry_comment = entry_comment

        self.label_count = label_count

        self.entry_class_0 = entry_class_0
        self.entry_class_1 = entry_class_1

        self.var_status = StringVar()
        self.var_status.set("Ready to load data.")  # Default status message

        self.label_status = Label(parent, textvariable=self.var_status, anchor="w", fg="white",
                                  bg="black")
        self.label_status.pack(side=tk.BOTTOM, fill=tk.X)

        return None

    def update_status(self, message: str) -> None:
        self.var_status.set(message)
        self.parent.update_idletasks()  # Refresh GUI immediately
        return None

    def _LayParameters(self, *args, **kwargs) -> None:

        parent = self.parent
        dy = self.dy

        reg = parent.register(self._Callback)

        label_0 = Label(parent, text="Radius:")
        label_0.place(height=30, width=128, x=144, y=1 * dy)

        var_0 = StringVar()

        entry_0 = Entry(parent, textvariable=var_0)
        entry_0.place(height=30, width=96, x=272, y=1 * dy)
        entry_0.config(state=tk.DISABLED, bg="Light grey")
        entry_0.config(validate="key", validatecommand=(reg, '%P'))

        label_aux0 = Label(parent, text='Å')
        label_aux0.place(height=30, width=16, x=368, y=1 * dy)

        label_1 = Label(parent, text="Aspect Ratio:")
        label_1.place(height=30, width=128, x=144, y=3 * dy)

        var_1 = StringVar()

        entry_1 = Entry(parent, textvariable=var_1)
        entry_1.place(height=30, width=96, x=272, y=3 * dy)
        entry_1.config(state=tk.DISABLED, bg="Light grey")
        entry_1.config(validate="key", validatecommand=(reg, '%P'))

        label_aux1 = Label(parent, text='%', justify="left")
        label_aux1.place(height=30, width=16, x=368, y=3 * dy)

        label_2 = Label(parent, text="PDI:")
        label_2.place(height=30, width=128, x=144, y=5 * dy)

        var_2 = StringVar()

        entry_2 = Entry(parent, textvariable=var_2)
        entry_2.place(height=30, width=96, x=272, y=5 * dy)
        entry_2.config(state=tk.DISABLED, bg="Light grey")
        entry_2.config(validate="key", validatecommand=(reg, '%P'))

        label_3 = Label(parent, text="Core Fraction:")
        label_3.place(height=30, width=128, x=144, y=7 * dy)

        var_3 = StringVar()

        entry_3 = Entry(parent, textvariable=var_3)
        entry_3.place(height=30, width=96, x=272, y=7 * dy)
        entry_3.config(state=tk.DISABLED, bg="Light grey")
        entry_3.config(validate="key", validatecommand=(reg, '%P'))

        label_aux2 = Label(parent, text='%')
        label_aux2.place(height=30, width=16, x=368, y=7 * dy)

        label_4 = Label(parent, text="Scattering Fraction:")
        label_4.place(height=30, width=128, x=144, y=9 * dy)

        var_4 = StringVar()

        entry_4 = Entry(parent, textvariable=var_4)
        entry_4.place(height=30, width=96, x=272, y=9 * dy)
        entry_4.config(state=tk.DISABLED, bg="Light grey")
        entry_4.config(validate="key", validatecommand=(reg, '%P'))

        label_aux3 = Label(parent, text='‰')
        label_aux3.place(height=30, width=16, x=368, y=9 * dy)

        label_5 = Label(parent, text="Corona Length:")
        label_5.place(height=30, width=128, x=144, y=11 * dy)

        var_5 = StringVar()

        entry_5 = Entry(parent, textvariable=var_5)
        entry_5.place(height=30, width=96, x=272, y=11 * dy)
        entry_5.config(state=tk.DISABLED, bg="Light grey")
        entry_5.config(validate="key", validatecommand=(reg, '%P'))

        label_aux4 = Label(parent, text='Å')
        label_aux4.place(height=30, width=16, x=368, y=11 * dy)

        label_6 = Label(parent, text="Core Density:")
        label_6.place(height=30, width=128, x=144, y=13 * dy)

        var_6 = StringVar()

        entry_6 = Entry(parent, textvariable=var_6)
        entry_6.place(height=30, width=96, x=272, y=13 * dy)
        entry_6.config(state=tk.DISABLED, bg="Light grey")
        entry_6.config(validate="key", validatecommand=(reg, '%P'))

        label_7 = Label(parent, text="Corona Density:")
        label_7.place(height=30, width=128, x=144, y=15 * dy)

        var_7 = StringVar()

        entry_7 = Entry(parent, textvariable=var_7)
        entry_7.place(height=30, width=96, x=272, y=15 * dy)
        entry_7.config(state=tk.DISABLED, bg="Light grey")
        entry_7.config(validate="key", validatecommand=(reg, '%P'))

        label_8 = Label(parent, text=r'R_g (ML)')
        label_8.place(height=30, width=96, x=680, y=1 * dy) 

        var_8 = StringVar()

        entry_8 = Entry(parent, textvariable=var_8)
        entry_8.place(height=30, width=96, x=680, y=3 * dy)
        entry_8.config(state=tk.DISABLED, bg="Light grey")
        entry_8.config(validate="key", validatecommand=(reg, '%P'))

        label_9 = Label(parent, text=r'R_g (GN)')
        label_9.place(height=30, width=96, x=680, y=5 * dy)

        var_9 = StringVar()

        entry_9 = Entry(parent, textvariable=var_9)
        entry_9.place(height=30, width=96, x=680, y=7 * dy)
        entry_9.config(state=tk.DISABLED, bg="Light grey")
        entry_9.config(validate="key", validatecommand=(reg, '%P'))

        self.entry_0 = entry_0
        self.entry_1 = entry_1
        self.entry_2 = entry_2
        self.entry_3 = entry_3
        self.entry_4 = entry_4
        self.entry_5 = entry_5
        self.entry_6 = entry_6
        self.entry_7 = entry_7
        self.entry_8 = entry_8
        self.entry_9 = entry_9

        self.parameter_labels = (
            label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7
        )
        self.parameter_aux_labels = {
            0: label_aux0,
            1: label_aux1,
            3: label_aux2,
            4: label_aux3,
            5: label_aux4,
        }

        self.label_1 = label_1
        self.label_aux1 = label_aux1

        return None

    def _LayButtons(self, *args, **kwargs) -> None:

        parent = self.parent
        dy = self.dy

        button_0_P_L = Button(parent, text='+', command=lambda: self._Change(0, 0))
        button_0_P_L.place(height=30, width=30, x=384, y=1 * dy)
        button_0_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_0_N_L = Button(parent, text='-', command=lambda: self._Change(0, 1))
        button_0_N_L.place(height=30, width=30, x=416, y=1 * dy)
        button_0_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_0_P_S = Button(parent, text='+', command=lambda: self._Change(0, 2))
        button_0_P_S.place(height=20, width=20, x=448, y=1 * dy + 5)
        button_0_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_0_N_S = Button(parent, text='-', command=lambda: self._Change(0, 3))
        button_0_N_S.place(height=20, width=20, x=480, y=1 * dy + 5)
        button_0_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_1_P_L = Button(parent, text='+', command=lambda: self._Change(1, 0))
        button_1_P_L.place(height=30, width=30, x=384, y=3 * dy)
        button_1_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_1_N_L = Button(parent, text='-', command=lambda: self._Change(1, 1))
        button_1_N_L.place(height=30, width=30, x=416, y=3 * dy)
        button_1_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_1_P_S = Button(parent, text='+', command=lambda: self._Change(1, 2))
        button_1_P_S.place(height=20, width=20, x=448, y=3 * dy + 5)
        button_1_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_1_N_S = Button(parent, text='-', command=lambda: self._Change(1, 3))
        button_1_N_S.place(height=20, width=20, x=480, y=3 * dy + 5)
        button_1_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_2_P_L = Button(parent, text='+', command=lambda: self._Change(2, 0))
        button_2_P_L.place(height=30, width=30, x=384, y=5 * dy)
        button_2_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_2_N_L = Button(parent, text='-', command=lambda: self._Change(2, 1))
        button_2_N_L.place(height=30, width=30, x=416, y=5 * dy)
        button_2_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_2_P_S = Button(parent, text='+', command=lambda: self._Change(2, 2))
        button_2_P_S.place(height=20, width=20, x=448, y=5 * dy + 5)
        button_2_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_2_N_S = Button(parent, text='-', command=lambda: self._Change(2, 3))
        button_2_N_S.place(height=20, width=20, x=480, y=5 * dy + 5)
        button_2_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_3_P_L = Button(parent, text='+', command=lambda: self._Change(3, 0))
        button_3_P_L.place(height=30, width=30, x=384, y=7 * dy)
        button_3_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_3_N_L = Button(parent, text='-', command=lambda: self._Change(3, 1))
        button_3_N_L.place(height=30, width=30, x=416, y=7 * dy)
        button_3_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_3_P_S = Button(parent, text='+', command=lambda: self._Change(3, 2))
        button_3_P_S.place(height=20, width=20, x=448, y=7 * dy + 5)
        button_3_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_3_N_S = Button(parent, text='-', command=lambda: self._Change(3, 3))
        button_3_N_S.place(height=20, width=20, x=480, y=7 * dy + 5)
        button_3_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_4_P_L = Button(parent, text='+', command=lambda: self._Change(4, 0))
        button_4_P_L.place(height=30, width=30, x=384, y=9 * dy)
        button_4_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_4_N_L = Button(parent, text='-', command=lambda: self._Change(4, 1))
        button_4_N_L.place(height=30, width=30, x=416, y=9 * dy)
        button_4_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_4_P_S = Button(parent, text='+', command=lambda: self._Change(4, 2))
        button_4_P_S.place(height=20, width=20, x=448, y=9 * dy + 5)
        button_4_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_4_N_S = Button(parent, text='-', command=lambda: self._Change(4, 3))
        button_4_N_S.place(height=20, width=20, x=480, y=9 * dy + 5)
        button_4_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_5_P_L = Button(parent, text='+', command=lambda: self._Change(5, 0))
        button_5_P_L.place(height=30, width=30, x=384, y=11 * dy)
        button_5_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_5_N_L = Button(parent, text='-', command=lambda: self._Change(5, 1))
        button_5_N_L.place(height=30, width=30, x=416, y=11 * dy)
        button_5_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_5_P_S = Button(parent, text='+', command=lambda: self._Change(5, 2))
        button_5_P_S.place(height=20, width=20, x=448, y=11 * dy + 5)
        button_5_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_5_N_S = Button(parent, text='-', command=lambda: self._Change(5, 3))
        button_5_N_S.place(height=20, width=20, x=480, y=11 * dy + 5)
        button_5_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_6_P_L = Button(parent, text='+', command=lambda: self._Change(6, 0))
        button_6_P_L.place(height=30, width=30, x=384, y=13 * dy)
        button_6_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_6_N_L = Button(parent, text='-', command=lambda: self._Change(6, 1))
        button_6_N_L.place(height=30, width=30, x=416, y=13 * dy)
        button_6_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_6_P_S = Button(parent, text='+', command=lambda: self._Change(6, 2))
        button_6_P_S.place(height=20, width=20, x=448, y=13 * dy + 5)
        button_6_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_6_N_S = Button(parent, text='-', command=lambda: self._Change(6, 3))
        button_6_N_S.place(height=20, width=20, x=480, y=13 * dy + 5)
        button_6_N_S.config(state=tk.DISABLED, bg="Light grey")

        button_7_P_L = Button(parent, text='+', command=lambda: self._Change(7, 0))
        button_7_P_L.place(height=30, width=30, x=384, y=15 * dy)
        button_7_P_L.config(state=tk.DISABLED, bg="Light grey")

        button_7_N_L = Button(parent, text='-', command=lambda: self._Change(7, 1))
        button_7_N_L.place(height=30, width=30, x=416, y=15 * dy)
        button_7_N_L.config(state=tk.DISABLED, bg="Light grey")

        button_7_P_S = Button(parent, text='+', command=lambda: self._Change(7, 2))
        button_7_P_S.place(height=20, width=20, x=448, y=15 * dy + 5)
        button_7_P_S.config(state=tk.DISABLED, bg="Light grey")

        button_7_N_S = Button(parent, text='-', command=lambda: self._Change(7, 3))
        button_7_N_S.place(height=20, width=20, x=480, y=15 * dy + 5)
        button_7_N_S.config(state=tk.DISABLED, bg="Light grey")

        self.button_0_P_L = button_0_P_L
        self.button_0_N_L = button_0_N_L
        self.button_0_P_S = button_0_P_S
        self.button_0_N_S = button_0_N_S
        self.button_1_P_L = button_1_P_L
        self.button_1_N_L = button_1_N_L
        self.button_1_P_S = button_1_P_S
        self.button_1_N_S = button_1_N_S
        self.button_2_P_L = button_2_P_L
        self.button_2_N_L = button_2_N_L
        self.button_2_P_S = button_2_P_S
        self.button_2_N_S = button_2_N_S
        self.button_3_P_L = button_3_P_L
        self.button_3_N_L = button_3_N_L
        self.button_3_P_S = button_3_P_S
        self.button_3_N_S = button_3_N_S
        self.button_4_P_L = button_4_P_L
        self.button_4_N_L = button_4_N_L
        self.button_4_P_S = button_4_P_S
        self.button_4_N_S = button_4_N_S
        self.button_5_P_L = button_5_P_L
        self.button_5_N_L = button_5_N_L
        self.button_5_P_S = button_5_P_S
        self.button_5_N_S = button_5_N_S
        self.button_6_P_L = button_6_P_L
        self.button_6_N_L = button_6_N_L
        self.button_6_P_S = button_6_P_S
        self.button_6_N_S = button_6_N_S
        self.button_7_P_L = button_7_P_L
        self.button_7_N_L = button_7_N_L
        self.button_7_P_S = button_7_P_S
        self.button_7_N_S = button_7_N_S

        return None

    def _LayPlots(self, *args, **kwargs) -> None:

        parent = self.parent

        # INCREASED DPI for sharper text, adjusted layout margins to prevent cutoff
        figure_s = Figure(figsize=(6, 3.4), dpi=100) 
        figure_s.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.90) 

        plot_s = figure_s.add_subplot(1, 1, 1)
        plot_s.set_title("Loaded Sample")
        plot_s.set_xlabel(r'q ($\AA^{-1}$)')
        plot_s.set_ylabel('Normalized Scattering Intensity')
        plot_s.set_xscale('log')
        plot_s.set_yscale('log')

        canvas_s = FigureCanvasTkAgg(figure_s, parent)
        
        # DYNAMIC SCALING: Use relwidth and relheight so the plot stretches with the window
        canvas_s.get_tk_widget().place(x=16, y=340, relwidth=0.96, relheight=0.50)

        self.figure_s = figure_s
        self.plot_s = plot_s
        self.canvas_s = canvas_s
        return None

    def _Layout(self, *args, **kwargs) -> None:

        """
        1. Set the parent window.
        2. Set the basic controls.
        3. Set the outputs for the parameters.
        4. Set the buttons for the parameters.
        5. Set the plots.
        """

        parent = self.parent

        tk.Frame.__init__(self, parent)

        parent.title('DLENS')
        
        # INCREASED WINDOW SIZE to fit all elements properly
        parent.geometry("860x760")
        parent.minsize(860, 760)
        parent.protocol("WM_DELETE_WINDOW", parent.quit)

        self.dy = 16

        self._LayOperation()
        self._LayParameters()
        self._LayButtons()
        self._LayPlots()

        return None