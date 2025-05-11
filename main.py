import os
import shutil
import sys

import numpy as np
import tkinter as tk
import tensorflow as tf
import pickle as pk

from datetime import datetime

from tkinter import Button, StringVar, filedialog, Entry, Label, OptionMenu, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from math import sqrt

from scaled_Debye import Disperse_Spheroid_Shell, Disperse_Cylinder_Shell

"""
The Debye module is a file that works alongside this file.
"""
from PIL import Image, ImageTk


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#FFFFE0", fg="#000000", relief="solid",
                         borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs) -> None:

        self.parent = parent

        self._Setting()
        self._Layout()
        self._LoadModels()

        return None

    def _Setting(self, *args, **kwargs) -> None:

        """
        1. Set the reference q-vector array.
        2. Set the current working directory.
        3. Set the base path.
        4. Set the working directory and log.
        5. Set the default values for the internal parameters.
        """

        # This is the standardized q values used for the inference.
        q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
        q_arr = np.power(10, q_log_arr - 2 * np.log10(2))

        self.q_log_arr = q_log_arr
        self.q_arr = q_arr

        cwd = os.getcwd()
        self.cwd = cwd

        """
        cwd: current working directory
            - base_path: base directory to store all runs
                - working_dir: directory for a new run
                    - This stores the exported SAXS data.
                - Also stores all the logs.

        """

        base_path = os.path.join(cwd, 'SAXS_SIM')

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        username = os.getlogin()
        current = datetime.now()
        current = current.strftime('%Y%m%d')

        count = 0

        while True:
            temp = f'{username}_{current}_{count}'
            if temp not in os.listdir(base_path):
                break
            else:
                count += 1

        log_end = 'csv'
        log_file = f'{temp}.{log_end}'

        working_dir = os.path.join(base_path, temp)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        log_path = os.path.join(base_path, log_file)

        with open(log_path, 'w') as f:
            f.write(
                'CWD,From,File,Shape,Param 1,Param 2,Param 3,Param 4,Param 5,Param 6,Param 7,mu 1,mu 2,sigma 1,sigma 2,'
                'Error,R_g,R_g,Comment\n'
            )

        self.log_file = log_file
        self.base_path = base_path
        self.working_dir = working_dir
        self.log_path = log_path

        """
        Parameter Descriptions:
            - file_loaded: return True if a file is loaded.
            - folder_loaded: return True if a folder is loaded.
            - fitted: return True if a simulation is ran.
            - started: return True if the user hit Start.

            - shape: a string value that describes the shape
            - _class: an integer value corresponding to shape
            - count: the file count of the folder loaded
        """

        self.file_loaded = False
        self.folder_loaded = False
        self.fitted = False
        self.started = False

        self.shape = None
        self._class = None
        self.count = -1

        return None

    def _LayOperation(self, *args, **kwargs) -> None:

        parent = self.parent
        dy = self.dy

        button_folder = Button(parent, text="Load Folder", command=self._LoadFolder)
        button_folder.place(height=2 * dy, width=128, x=16, y=1 * dy)
        ToolTip(button_folder, "Click to load a folder")

        button_file = Button(parent, text="Load File", command=self._LoadFile)
        button_file.place(height=2 * dy, width=128, x=16, y=3 * dy)
        ToolTip(button_file, "Click to load a file")

        shapes = ["Spheroid", "Cylinder"]
        select_shapes = StringVar()
        select_shapes.set("Fitting Algorithm")
        select_shapes.trace_add("write", self._Change_Mode)

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

        label_count = Label(parent, text="N/A")
        label_count.place(height=30, width=64, x=16, y=17 * dy)

        button_backward = Button(parent, text='<', command=self._Backward)
        button_backward.place(height=30, width=30, x=80, y=17 * dy)
        button_backward.config(state=tk.DISABLED, bg="Light grey")

        button_forward = Button(parent, text='>', command=self._Forward)
        button_forward.place(height=30, width=30, x=112, y=17 * dy)
        button_forward.config(state=tk.DISABLED, bg="Light grey")

        label_MSLE = Label(parent, text="mMSLE:")
        label_MSLE.place(height=30, width=128, x=144, y=17 * dy)
        ToolTip(label_MSLE, "Displays the Mean Squared Logarithmic Error")

        var_MSLE = StringVar()

        Entry_MSLE = Entry(parent, textvariable=var_MSLE)
        Entry_MSLE.place(height=30, width=96, x=272, y=17 * dy)
        Entry_MSLE.config(state=tk.DISABLED, bg="Light grey")

        label_comment = Label(parent, text="Comment:")
        label_comment.place(height=30, width=128, x=16, y=19 * dy)

        var_comment = StringVar()

        entry_comment = Entry(parent, textvariable=var_comment)
        entry_comment.place(height=30, width=192, x=144, y=19 * dy)
        entry_comment.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

        label_class_0 = Label(parent, text="Spheroid:")
        label_class_0.place(height=30, width=64, x=336, y=19 * dy)

        var__class0 = StringVar()

        entry_class_0 = Entry(parent, textvariable=var__class0)
        entry_class_0.place(height=30, width=64, x=400, y=19 * dy)
        entry_class_0.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(entry_class_0, "Probability of being a Spheroid")

        label_class_1 = Label(parent, text="Cylinder:")
        label_class_1.place(height=30, width=64, x=464, y=19 * dy)
        ToolTip(label_class_1, "Probability of being a Cylinder")

        var__class1 = StringVar()

        entry_class_1 = Entry(parent, textvariable=var__class1)
        entry_class_1.place(height=30, width=64, x=528, y=19 * dy)
        entry_class_1.config(state=tk.DISABLED, bg="Light grey")

        button_autosub = Button(parent, text='Auto-subtraction', command=self._Autosubtraction)
        button_autosub.place(height=30, width=110, x=512, y=9 * dy)
        ToolTip(button_autosub, "Click to perform auto-subtraction")

        button_help = Button(parent, text='Help', command=self._Help)
        button_help.place(height=30, width=110, x=512, y=11 * dy)
        ToolTip(button_help, "Click to access help")
        self.drop_methods = drop_methods

        self.select_shapes = select_shapes

        self.button_file = button_file
        self.button_folder = button_folder
        self.button_clear = button_clear
        self.button_export = button_export
        self.button_simulate = button_simulate
        self.button_visualize = button_visualize
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
        label_8.place(height=30, width=96, x=512, y=1 * dy)

        var_8 = StringVar()

        entry_8 = Entry(parent, textvariable=var_8)
        entry_8.place(height=30, width=96, x=512, y=3 * dy)
        entry_8.config(state=tk.DISABLED, bg="Light grey")
        entry_8.config(validate="key", validatecommand=(reg, '%P'))

        label_9 = Label(parent, text=r'R_g (GN)')
        label_9.place(height=30, width=96, x=512, y=5 * dy)

        var_9 = StringVar()

        entry_9 = Entry(parent, textvariable=var_9)
        entry_9.place(height=30, width=96, x=512, y=7 * dy)
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

    def resource_path(self, relative_path):
        """
        Get absolute path to resource, works for dev and for PyInstaller one-file bundle.
        """
        if getattr(sys, 'frozen', False):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, relative_path)

    def _LayPlots(self, *args, **kwargs) -> None:

        parent = self.parent
        dy = self.dy

        reg = parent.register(self._Callback)

        figure_s = Figure(figsize=(5, 3), dpi=64)

        plot_s = figure_s.add_subplot(1, 1, 1)
        plot_s.set_title("Loaded Sample")
        plot_s.set_xlabel(r'R ($\AA$)')
        plot_s.set_ylabel('Normalized Scattering Intensity')
        plot_s.set_xscale('log')
        plot_s.set_yscale('log')

        canvas_s = FigureCanvasTkAgg(figure_s, parent)
        canvas_s.get_tk_widget().place(height=336, width=560, x=640, y=0 * dy)

        figure_0 = Figure(figsize=(4, 3), dpi=64)

        plot_0 = figure_0.add_subplot(1, 1, 1)
        plot_0.set_title("Radius Probability Distribution")
        plot_0.set_xlabel(r'Radius ($\AA$)')
        plot_0.set_ylabel('Probability Density')

        canvas_0 = FigureCanvasTkAgg(figure_0, parent)
        canvas_0.get_tk_widget().place(height=300, width=400, x=16, y=22 * dy)

        label_0_m = Label(parent, text="Mean (Å):")
        label_0_m.place(height=30, width=64, x=16, y=41 * dy)

        var_0_m = StringVar()

        entry_0_m = Entry(parent, textvariable=var_0_m)
        entry_0_m.place(height=30, width=64, x=80, y=41 * dy)
        entry_0_m.config(state=tk.DISABLED, bg="Light grey")
        entry_0_m.config(validate="key", validatecommand=(reg, '%P'))

        label_0_s = Label(parent, text="STD (Å):")
        label_0_s.place(height=30, width=64, x=144, y=41 * dy)

        var_0_s = StringVar()

        entry_0_s = Entry(parent, textvariable=var_0_s)
        entry_0_s.place(height=30, width=64, x=208, y=41 * dy)
        entry_0_s.config(state=tk.DISABLED, bg="Light grey")
        entry_0_s.config(validate="key", validatecommand=(reg, '%P'))

        label_0_d = Label(parent, text="Deviation:", justify="left")
        label_0_d.place(height=30, width=64, x=272, y=41 * dy)

        var_0_d = StringVar()

        entry_0_d = Entry(parent, textvariable=var_0_d)
        entry_0_d.place(height=30, width=64, x=336, y=41 * dy)
        entry_0_d.config(state=tk.DISABLED, bg="Light grey")
        entry_0_d.config(validate="key", validatecommand=(reg, '%P'))

        figure_1 = Figure(figsize=(4, 3), dpi=64)

        plot_1 = figure_1.add_subplot(1, 1, 1)
        plot_1.set_title("Aspect Ratio Probability Distribution")
        plot_1.set_xlabel(r'Aspect ratio (%)')
        plot_1.set_ylabel('Probability Density')

        canvas_1 = FigureCanvasTkAgg(figure_1, parent)
        canvas_1.get_tk_widget().place(height=300, width=400, x=432, y=22 * dy)

        label_1_m = Label(parent, text="Mean (%):")
        label_1_m.place(height=30, width=64, x=432, y=41 * dy)

        var_1_m = StringVar()

        entry_1_m = Entry(parent, textvariable=var_1_m)
        entry_1_m.place(height=30, width=64, x=496, y=41 * dy)
        entry_1_m.config(state=tk.DISABLED, bg="Light grey")
        entry_1_m.config(validate="key", validatecommand=(reg, '%P'))

        label_1_s = Label(parent, text="STD (%):")
        label_1_s.place(height=30, width=64, x=560, y=41 * dy)

        var_1_s = StringVar()

        entry_1_s = Entry(parent, textvariable=var_1_s)
        entry_1_s.place(height=30, width=64, x=624, y=41 * dy)
        entry_1_s.config(state=tk.DISABLED, bg="Light grey")
        entry_1_s.config(validate="key", validatecommand=(reg, '%P'))

        label_1_d = Label(parent, text="Deviation:")
        label_1_d.place(height=30, width=64, x=688, y=41 * dy)

        var_1_d = StringVar()

        entry_1_d = Entry(parent, textvariable=var_1_d)
        entry_1_d.place(height=30, width=64, x=752, y=41 * dy)
        entry_1_d.config(state=tk.DISABLED, bg="Light grey")
        entry_1_d.config(validate="key", validatecommand=(reg, '%P'))

        figure_2 = Figure(figsize=(4, 3), dpi=64)

        plot_2 = figure_2.add_subplot(1, 1, 1)
        plot_2.set_title("PDI Probability Distribution")
        plot_2.set_xlabel(r'PDI')
        plot_2.set_xscale('log')
        plot_2.set_ylabel('Probability Density')

        canvas_2 = FigureCanvasTkAgg(figure_2, parent)
        canvas_2.get_tk_widget().place(height=300, width=400, x=864, y=22 * dy)

        label_2_m = Label(parent, text="Mean:")
        label_2_m.place(height=30, width=64, x=864, y=41 * dy)

        var_2_m = StringVar()

        entry_2_m = Entry(parent, textvariable=var_2_m)
        entry_2_m.place(height=30, width=64, x=928, y=41 * dy)
        entry_2_m.config(state=tk.DISABLED, bg="Light grey")
        entry_2_m.config(validate="key", validatecommand=(reg, '%P'))

        label_2_s = Label(parent, text="STD:")
        label_2_s.place(height=30, width=64, x=992, y=41 * dy)

        var_2_s = StringVar()

        entry_2_s = Entry(parent, textvariable=var_2_s)
        entry_2_s.place(height=30, width=64, x=1056, y=41 * dy)
        entry_2_s.config(state=tk.DISABLED, bg="Light grey")
        entry_2_s.config(validate="key", validatecommand=(reg, '%P'))

        label_2_d = Label(parent, text="Deviation:")
        label_2_d.place(height=30, width=64, x=1120, y=41 * dy)

        var_2_d = StringVar()

        entry_2_d = Entry(parent, textvariable=var_2_d)
        entry_2_d.place(height=30, width=64, x=1184, y=41 * dy)
        entry_2_d.config(state=tk.DISABLED, bg="Light grey")
        entry_2_d.config(validate="key", validatecommand=(reg, '%P'))

        # 1. Use resource_path to locate the file
        image_path = self.resource_path("horizontal_logo.png")

        # 2. Open, resize, and convert to a Tk image
        orig_image = Image.open(image_path)
        resized_image = orig_image.resize((400, 110), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)

        # 3. Create a Label and keep a reference to the PhotoImage
        image_label = Label(parent, image=tk_image)
        image_label.image = tk_image  # prevent GC

        # 4. Place it
        image_label.place(x=432, y=700, width=400, height=96)

        self.entry_0_m = entry_0_m
        self.entry_0_s = entry_0_s
        self.entry_0_d = entry_0_d
        self.entry_1_m = entry_1_m
        self.entry_1_s = entry_1_s
        self.entry_1_d = entry_1_d
        self.entry_2_m = entry_2_m
        self.entry_2_s = entry_2_s
        self.entry_2_d = entry_2_d

        self.label_1_m = label_1_m
        self.label_1_s = label_1_s

        self.figure_s = figure_s
        self.figure_0 = figure_0
        self.figure_1 = figure_1
        self.figure_2 = figure_2

        self.plot_s = plot_s
        self.plot_0 = plot_0
        self.plot_1 = plot_1
        self.plot_2 = plot_2

        self.canvas_s = canvas_s
        self.canvas_0 = canvas_0
        self.canvas_1 = canvas_1
        self.canvas_2 = canvas_2

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

        parent.title('SAXS SIM')
        parent.geometry("1280x720")
        parent.protocol("WM_DELETE_WINDOW", parent.quit())

        self.dy = 16

        self._LayOperation()
        self._LayParameters()
        self._LayButtons()
        self._LayPlots()
        return None

    def _LoadModels(self, *args, **kwargs) -> None:

        if getattr(sys, 'frozen', False):  # App is frozen (e.g., bundled by pyinstaller)
            app_dir = os.path.dirname(os.path.abspath(sys.executable))
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(app_dir)))  # go from D_Lens.app/Contents/MacOS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        model_dir = os.path.join(base_dir, 'Models')

        name_s_0 = "2025_01_21_sphere_CPNN_Radius_0.keras"
        name_s_1 = "2025_01_21_sphere_CPNN_AspectRatio_0.keras"
        name_s_2 = '2025_03_09_sphere_CPNN_PDI_0.keras'
        name_s_3 = '2025_01_21_sphere_CPNN_GyrationRadius_0.keras'

        name_c_0 = "2025_01_21_cylinder_CPNN_Radius_0.keras"
        name_c_1 = "2025_01_21_cylinder_CPNN_AspectRatio_0.keras"
        name_c_2 = '2025_03_09_cylinder_CPNN_PDI_0.keras'
        name_c_3 = '2025_01_21_cylinder_CPNN_GyrationRadius_0.keras'

        name_qr = "2025_01_26_SCNN_qr_0.keras"

        name_cl = "2025_01_26_SVM_C_0.pkl"

        path_s_0 = os.path.join(model_dir, name_s_0)
        path_s_1 = os.path.join(model_dir, name_s_1)
        path_s_2 = os.path.join(model_dir, name_s_2)
        path_s_3 = os.path.join(model_dir, name_s_3)

        path_c_0 = os.path.join(model_dir, name_c_0)
        path_c_1 = os.path.join(model_dir, name_c_1)
        path_c_2 = os.path.join(model_dir, name_c_2)
        path_c_3 = os.path.join(model_dir, name_c_3)

        path_qr = os.path.join(model_dir, name_qr)

        path_cl = os.path.join(model_dir, name_cl)

        model_s_0 = tf.keras.models.load_model(path_s_0, compile=False)
        model_s_1 = tf.keras.models.load_model(path_s_1, compile=False)
        model_s_2 = tf.keras.models.load_model(path_s_2, compile=False)
        model_s_3 = tf.keras.models.load_model(path_s_3, compile=False)

        model_c_0 = tf.keras.models.load_model(path_c_0, compile=False)
        model_c_1 = tf.keras.models.load_model(path_c_1, compile=False)
        model_c_2 = tf.keras.models.load_model(path_c_2, compile=False)
        model_c_3 = tf.keras.models.load_model(path_c_3, compile=False)

        model_qr = tf.keras.models.load_model(path_qr, compile=False)

        with open(path_cl, 'rb') as f:
            model_cl = pk.load(f)

        self.model_s_0 = model_s_0
        self.model_s_1 = model_s_1
        self.model_s_2 = model_s_2
        self.model_s_3 = model_s_3

        self.model_c_0 = model_c_0
        self.model_c_1 = model_c_1
        self.model_c_2 = model_c_2
        self.model_c_3 = model_c_3

        self.model_qr = model_qr

        self.model_cl = model_cl

        return None

    def _Pop_Up_0(self, *args, **kwargs) -> None:

        """
        Start the auto-subtraction window.
        """

        self.pop = tk.Toplevel()
        self._Set_Sub_UI()

        return None

    def _Set_Sub_UI(self, *args, **kwargs) -> None:

        """
        1. Set the base parameters.
        2. Set the working folders.
        3. Set the buttons.
        4. Set the plots.
        """

        parent = self.pop

        parent.title('Autosubtraction')
        parent.geometry("910x400")

        self._Set_Sub_Numbers()
        self._Set_Sub_Folders()
        self._Set_Sub_Buttons()
        self._Set_Sub_Plots()
        self.status_var = tk.StringVar()
        self.status_var.set("Waiting for input...")

        status_frame = tk.Frame(parent, height=25, bg="white")
        status_frame.pack(side="bottom", fill="x")

        self.sub_status_var = tk.StringVar()
        self.sub_status_var.set("Ready to load data.")

        self.status_label = tk.Label(parent, text="Ready", anchor="w", relief=tk.SUNKEN, bg="white",
                                     fg="black")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = self.status_label

        return None

    def _Set_Sub_Buttons(self, *args, **kwargs) -> None:

        parent = self.pop
        dy = self.dy
        reg = parent.register(self._Callback)

        height = 3 * dy - 2
        dx = 8
        width = 160

        button_raw = Button(parent, text="Load Raw Data", command=self._Sub_Load_0)
        button_raw.place(height=height, width=width, x=dx, y=1 * dy)
        button_raw.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        ToolTip(button_raw, "Load the raw data file for subtraction.")

        button_back = Button(parent, text="Load Buffer Data", command=self._Sub_Load_1)
        button_back.place(height=height, width=width, x=dx, y=4 * dy)
        button_back.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        ToolTip(button_back, "Load the buffer data file for subtraction.")

        button_sub = Button(parent, text="Subtract", command=self._Sub_Subtract)
        button_sub.place(height=height, width=width, x=dx, y=7 * dy)
        button_sub.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_sub, "Perform subtraction between raw and buffer data.")

        button_clear = Button(parent, text="Clear", command=self._Sub_Clear)
        button_clear.place(height=height, width=width, x=dx, y=10 * dy)
        button_clear.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_clear, "Clear all loaded data and reset settings.")

        button_use = Button(parent, text="Use", command=self._Sub_Use)
        button_use.place(height=height, width=width, x=dx, y=13 * dy)
        button_use.config(state=tk.DISABLED, bg="Light grey")
        ToolTip(button_use, "Use the subtracted data in D-Lens.")

        button_cutoff = Button(parent, text="q-Cutoff", command=self._Sub_Cutoff)
        button_cutoff.place(height=height, width=width / 2, x=dx, y=16 * dy)
        ToolTip(button_cutoff, "Set the q-value cutoff for data subtraction.")

        var_q = StringVar()

        entry_q = Entry(parent, textvariable=var_q)
        entry_q.place(height=height, width=width / 2, x=dx + width / 2, y=16 * dy)
        entry_q.config(validate="key", validatecommand=(reg, '%P'))
        entry_q.insert(0, f'{self.q_crit}')

        label_comment = Label(parent, text='Comments')
        label_comment.place(height=30, width=width, x=dx, y=19 * dy)

        var_comment = StringVar()

        entry_comment = Entry(parent, textvariable=var_comment)
        entry_comment.place(height=height, width=width, x=dx, y=21 * dy)

        self.sub_button_raw = button_raw
        self.sub_button_back = button_back
        self.sub_button_sub = button_sub
        self.sub_button_clear = button_clear
        self.sub_button_use = button_use

        self.sub_var_q = var_q
        self.sub_var_comment = var_comment

        return None

    def _Set_Sub_Plots(self, *args, **kwargs) -> None:

        parent = self.pop

        figure_0 = Figure(figsize=(4, 4), dpi=64)

        plot_0 = figure_0.add_subplot(1, 1, 1)
        plot_0.set_title("Raw")
        plot_0.set_xlabel(r'q ($\AA^{-1}$)')
        plot_0.set_ylabel('Scattering Intensity')
        plot_0.set_xscale('log')
        plot_0.set_yscale('log')

        canvas_0 = FigureCanvasTkAgg(figure_0, parent)
        canvas_0.get_tk_widget().place(height=360, width=360, x=180, y=20)

        figure_1 = Figure(figsize=(4, 4), dpi=64)

        plot_1 = figure_1.add_subplot(1, 1, 1)
        plot_1.set_title("Subtracted")
        plot_1.set_xlabel(r'q ($\AA^{-1}$)')
        plot_1.set_ylabel('Scattering Intensity')
        plot_1.set_xscale('log')
        plot_1.set_yscale('log')

        canvas_1 = FigureCanvasTkAgg(figure_1, parent)
        canvas_1.get_tk_widget().place(height=360, width=360, x=540, y=20)

        self.sub_figure_0 = figure_0
        self.sub_figure_1 = figure_1

        self.sub_plot_0 = plot_0
        self.sub_plot_1 = plot_1

        self.sub_canvas_0 = canvas_0
        self.sub_canvas_1 = canvas_1

        return None

    def _Set_Sub_Numbers(self, *args, **kwargs) -> None:

        q_log_arr = np.arange(-2.0, 0.0, np.true_divide(1, 128) - 2 * np.log10(2))
        q_arr = np.power(10, q_log_arr)

        self.sub_q_arr = q_arr
        self.q_crit = 0.2

        self.sub_loaded_0 = False
        self.sub_loaded_1 = False
        self.sub_loaded_2 = False

        return None

    def _Set_Sub_Folders(self, *args, **kwargs) -> None:

        """
        This function creates the necessary paths and files.

        CWD
            - Subtraction (base_path)
                - (working_dir)
                    - Raw           (raw_dir)
                    - Background    (back_dir)
                    - Subtracted    (sub_dir)
                    - Images        (img_dir)
                    - (sub_log_path)
        """

        cwd = os.getcwd()
        username = os.getlogin()
        current = datetime.now()
        current = current.strftime('%Y%m%d')

        base_path = os.path.join(cwd, 'Subtraction')

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        count = 0

        while True:
            temp = f'{username}_{current}_{count:02d}'
            if temp not in os.listdir(base_path):
                break
            else:
                count += 1

        working_dir = os.path.join(base_path, temp)
        raw_dir = os.path.join(working_dir, 'Raw')
        back_dir = os.path.join(working_dir, 'Background')
        sub_dir = os.path.join(working_dir, 'Subtracted')
        img_dir = os.path.join(working_dir, 'Images')

        log_end = 'csv'
        log_file = f'Record.{log_end}'
        sub_log_path = os.path.join(working_dir, log_file)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        if not os.path.exists(back_dir):
            os.makedirs(back_dir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        with open(sub_log_path, "a") as f:
            f.write("Raw,Background,Subtracted,Scale factor,Minimum q value,Comments\n")

        self.sub_base_path = base_path
        self.sub_working_dir = working_dir
        self.sub_raw_dir = raw_dir
        self.sub_back_dir = back_dir
        self.sub_sub_dir = sub_dir
        self.sub_img_dir = img_dir
        self.sub_log_path = sub_log_path

        return None

    def _Sub_Clear(self) -> None:
        # Prevent accidental data loss
        if messagebox.askyesno("Confirm", "Clear all current data?"):
            self._Sub_Clear_Buttons()
            self._Sub_Clear_Variables()
            self._Sub_Clear_File()
        """
        1. Clear buttons.
        2. Clear variables.
        3. Clear file.
        """
        self.sub_status_var.set("Clearing loaded data...")
        self._Sub_Clear_Buttons()
        self._Sub_Clear_Variables()
        self._Sub_Clear_File()
        self.sub_status_var.set("Ready to load data.")

        return None

    def _Sub_Clear_Buttons(self) -> None:

        self.sub_button_raw.config(text='Raw')
        self.sub_button_back.config(text='Buffer')
        self.sub_button_sub.config(state=tk.DISABLED, bg="Light grey")
        self.sub_button_clear.config(state=tk.DISABLED, bg="Light grey")
        self.sub_button_use.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _Sub_Clear_Variables(self) -> None:

        self.sub_loaded_0 = False
        self.sub_loaded_1 = False
        self.sub_loaded_2 = False

        return None

    def _Sub_Clear_File(self) -> None:

        self.sub_plot_0.clear()
        self.sub_plot_0.set_title("Raw")
        self.sub_plot_0.set_xlabel(r'q ($\AA^{-1}$)')
        self.sub_plot_0.set_ylabel('Scattering Intensity')
        self.sub_plot_0.set_xscale('log')
        self.sub_plot_0.set_yscale('log')
        self.sub_plot_0.grid()

        self.sub_canvas_0.draw()

        self.sub_plot_1.clear()
        self.sub_plot_1.set_title("Subtracted")
        self.sub_plot_1.set_xlabel(r'q ($\AA^{-1}$)')
        self.sub_plot_1.set_ylabel('Scattering Intensity')
        self.sub_plot_1.set_xscale('log')
        self.sub_plot_1.set_yscale('log')
        self.sub_plot_1.grid()

        self.sub_canvas_1.draw()

        return None

    def _Sub_Use(self) -> None:
        self.sub_status_var.set("Saving and applying subtracted data...")
        """
        1. Check if the subtraction was performed.
        2. Fetch the raw, background, and subtracted data.
        3. Copy the raw and background data to the created folders.
        4. Save the subtracted data to the created folder.
        5. Clear the files.
        6. Set the subtracted data to the working data.
        7. Run the analysis on the working data.
        8. Quit the sub-window.
        """

        if self.sub_loaded_2:
            raw_origin = self.file_0
            back_origin = self.file_1
            sub_log_path = self.sub_log_path

            raw_dir = self.sub_raw_dir
            back_dir = self.sub_back_dir
            sub_dir = self.sub_sub_dir

            raw_short = os.path.basename(raw_origin)
            back_short = os.path.basename(raw_origin)
            sub_short = os.path.basename(raw_origin)

            raw_name, _ = raw_short.split('.')
            back_name, _ = back_short.split('.')
            sub_name, _ = sub_short.split('.')

            new_raw_name = raw_name + '.csv'
            new_back_name = back_name + '.csv'
            new_sub_name = sub_name + '_sub.scv'

            new_raw_path = os.path.join(raw_dir, new_raw_name)
            new_back_path = os.path.join(back_dir, new_back_name)
            new_sub_path = os.path.join(sub_dir, new_sub_name)

            alpha = self.alpha
            q_crit = self.q_crit
            comment = self.comment

            self._Clear()
            self.button_simulate.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

            with open(sub_log_path, "a") as f:
                f.write(f"{raw_origin},{back_origin},{new_sub_path},{alpha},{q_crit},{comment}\n")

            Is_0 = self.Is_0
            qs_0 = self.qs_0
            Is_1 = self.Is_1
            qs_1 = self.qs_1
            qs_2 = self.qs_2
            Is_2 = self.Is_2

            Is_2[Is_2 <= 0] = np.min(Is_2[Is_2 > 0])

            Is_2 = np.interp(self.q_arr, qs_2, Is_2)
            Is_2 /= np.max(Is_2)

            self.I_arr = Is_2

            temp_0 = np.hstack((Is_0.reshape(-1, 1), qs_0.reshape(-1, 1)))
            temp_1 = np.hstack((Is_1.reshape(-1, 1), qs_1.reshape(-1, 1)))
            temp_2 = np.hstack((Is_2.reshape(-1, 1), self.q_arr.reshape(-1, 1)))

            np.savetxt(new_raw_path, temp_0, delimiter=",")
            np.savetxt(new_back_path, temp_1, delimiter=",")
            np.savetxt(new_sub_path, temp_2, delimiter=",")

            self.file_loaded = True
            self.folder_loaded = False
            self.origin = sub_dir
            self.file_path = new_sub_path

            self._Draw_qI()
            self._Classify()
            self._Fit()
            self.sub_status_var.set("Data applied. Ready.")
            self.pop.destroy()

        return None

    def _Sub_Load_File(self) -> None:

        """
        1. Get the file.
        2. Get the file path.
        3. Prepare the file.
        """

        parent = self.pop

        parent.path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select a File"
        )

        filename = parent.path

        if filename:
            filenameshort = os.path.basename(filename)

            name_len = len(filenameshort)
            folder_name = filename[:-name_len]

            self.sub_working_origin = folder_name
            self.sub_working_file = filename

            self._Sub_Prepare_File()

        return None

    def _Sub_Load_0(self) -> None:

        """
        Load the raw file.
        """
        self.sub_status_var.set("Loading raw data...")
        self.working_index = 0
        self.sub_loaded_0 = True
        self._Sub_Load_File()
        self.sub_status_var.set("Raw data loaded.")

        return None

    def _Sub_Load_1(self) -> None:

        """
        Load the background file.
        """
        self.sub_status_var.set("Loading buffer data...")
        self.working_index = 1
        self.sub_loaded_1 = True
        self._Sub_Load_File()
        self.sub_status_var.set("Buffer data loaded.")

        return None

    def _Sub_Prepare_File(self) -> None:

        """
        1. Get the data from the working file.
        2. Store the data into the corresponding variables.
        3. Update the button labels.
        4. Update the plot.
        """

        self._Sub_get_qI()
        self._Sub_Update_Data()
        self._Sub_Update_Button()
        self._Sub_Update_Plot_0()

        return None

    def _Sub_Update_Data(self) -> None:

        # Assign the gathered data to the appropriate labels.

        if self.working_index == 0:
            self.qs_0 = self.working_qs
            self.Is_0 = self.working_Is
            self.ss_0 = self.working_ss
            self.origin_0 = self.sub_working_origin
            self.file_0 = self.sub_working_file
        elif self.working_index == 1:
            self.qs_1 = self.working_qs
            self.Is_1 = self.working_Is
            self.ss_1 = self.working_ss
            self.origin_1 = self.sub_working_origin
            self.file_1 = self.sub_working_file
        else:
            pass

        return None

    def _Sub_Update_Button(self) -> None:

        """
        1. Check whether the raw or background data is loaded.
        2. Update the buttons.
        """

        if self.working_index == 0:
            self.sub_button_raw.config(text=os.path.basename(self.sub_working_file))
        elif self.working_index == 1:
            self.sub_button_back.config(text=os.path.basename(self.sub_working_file))
        else:
            pass

        if self.sub_loaded_0 and self.sub_loaded_1:
            self.sub_button_sub.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.sub_button_clear.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.sub_button_use.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        elif self.sub_loaded_0 or self.sub_loaded_1:
            self.sub_button_sub.config(state=tk.DISABLED, bg="Light grey")
            self.sub_button_clear.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.sub_button_use.config(state=tk.DISABLED, bg="Light grey")
        else:
            self.sub_button_sub.config(state=tk.DISABLED, bg="Light grey")
            self.sub_button_clear.config(state=tk.DISABLED, bg="Light grey")
            self.sub_button_use.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _Sub_Update_Plot_0(self) -> None:

        if self.sub_loaded_0 and not self.sub_loaded_1:
            self.sub_plot_0.clear()
            self.sub_plot_0.plot(self.qs_0, self.Is_0)
            self.sub_plot_0.axvline(x=self.q_crit, color='r')
            self.sub_plot_0.set_title("Raw")
            self.sub_plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.sub_plot_0.set_ylabel('Scattering Intensity')
            self.sub_plot_0.set_xscale('log')
            self.sub_plot_0.set_yscale('log')
            self.sub_plot_0.grid()

            self.sub_canvas_0.draw()

        elif self.sub_loaded_1 and not self.sub_loaded_0:
            self.sub_plot_0.clear()
            self.sub_plot_0.plot(self.qs_1, self.Is_1)
            self.sub_plot_0.axvline(x=self.q_crit, color='r')
            self.sub_plot_0.set_title("Raw")
            self.sub_plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.sub_plot_0.set_ylabel('Scattering Intensity')
            self.sub_plot_0.set_xscale('log')
            self.sub_plot_0.set_yscale('log')
            self.sub_plot_0.grid()

            self.sub_canvas_0.draw()

        elif self.sub_loaded_0 and self.sub_loaded_1:
            self.sub_plot_0.clear()
            self.sub_plot_0.plot(self.qs_0, self.Is_0, label='Raw')
            self.sub_plot_0.plot(self.qs_1, self.Is_1, label='Background')
            self.sub_plot_0.axvline(x=self.q_crit, color='r')
            self.sub_plot_0.set_title("Raw")
            self.sub_plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.sub_plot_0.set_ylabel('Scattering Intensity')
            self.sub_plot_0.set_xscale('log')
            self.sub_plot_0.set_yscale('log')
            self.sub_plot_0.legend()
            self.sub_plot_0.grid()

            self.sub_canvas_0.draw()

        else:
            self.sub_plot_0.clear()
            self.sub_plot_0.set_title("Raw")
            self.sub_plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.sub_plot_0.set_ylabel('Scattering Intensity')
            self.sub_plot_0.set_xscale('log')
            self.sub_plot_0.set_yscale('log')
            self.sub_plot_0.legend()
            self.sub_plot_0.grid()

            self.sub_canvas_0.draw()

        return None

    def _Sub_Subtract(self) -> None:

        """
        1. Perform auto-subtraction.
        2. Update the plot.
        """
        self.sub_status_var.set("Performing subtraction...")
        self._Sub_Auto_Subtract()
        self._Sub_Update_Plot_1()
        self.sub_status_var.set("Subtraction complete.")

        return None

    def _Sub_Auto_Subtract(self) -> None:

        """
        1. Check if both the raw and background files are loaded.
        2. Get the critical q-value.
        3. Cut off the useful values.
        4. Perform least-square fit.
        5. Scale the background and subtract from the raw data.
        """

        q_crit = float(self.sub_var_q.get())
        self.q_crit = q_crit

        self.comment = self.sub_var_comment.get()

        if self.sub_loaded_0 and self.sub_loaded_1:
            temp_0 = self.Is_0[self.qs_0 > q_crit]
            temp_1 = self.Is_1[self.qs_1 > q_crit]

            sum_01 = np.sum(temp_0 * temp_1)
            sum_11 = np.sum(np.square(temp_1))

            alpha = sum_01 / sum_11
            self.alpha = alpha

            self.qs_2 = self.qs_0
            self.Is_2 = self.Is_0 - alpha * self.Is_1
            self.ss_2 = np.sqrt(np.square(self.ss_0) + np.square(alpha * self.ss_1))
            self.sub_loaded_2 = True

        return None

    def _Sub_Update_Plot_1(self) -> None:

        if self.sub_loaded_2:
            self.sub_plot_1.clear()
            self.sub_plot_1.plot(self.qs_2, self.Is_2)
            self.sub_plot_1.set_title("Subtracted")
            self.sub_plot_1.set_xlabel(r'q ($\AA^{-1}$)')
            self.sub_plot_1.set_ylabel('Scattering Intensity')
            self.sub_plot_1.set_xscale('log')
            self.sub_plot_1.set_yscale('log')
            self.sub_plot_1.grid()

            self.sub_canvas_1.draw()

        return None

    def _Sub_Cutoff(self) -> None:

        q_crit = float(self.sub_var_q.get())

        working_qs = self.working_qs

        if len(working_qs[working_qs >= q_crit]) < 8:
            q_crit = working_qs[-8]

        self.q_crit = q_crit
        self._Sub_Update_Plot_0()

        return None

    def _Sub_get_qI(self, *args, **kwargs) -> None:

        working_file = self.sub_working_file

        filenameshort = os.path.basename(working_file)
        end = filenameshort[-3:]

        if 'button' in kwargs:
            _button = kwargs['button']
            _button.configure(text=filenameshort)

        temp = list()

        with open(working_file, 'r') as f:

            while f.readline().lstrip()[0] not in "0123456789":
                continue

            while True:
                line = f.readline().lstrip()

                if line:
                    if end == 'csv':
                        num = len(line.split(','))
                    else:
                        num = len(line.split())

                    if num == 2:
                        q, I = line.split()
                        temp.append((float(q), float(I)))
                    else:
                        q, I, s = line.split()
                        temp.append((float(q), float(I), float(s)))

                else:
                    break

        temp = np.array(temp)

        qs = temp[:, 0]
        Is = temp[:, 1]

        if temp.shape[1] == 2:
            ss = np.sqrt(Is)
        else:
            ss = temp[:, 2]

        self.working_qs = qs
        self.working_Is = Is
        self.working_ss = ss

        return None

    def _Pop_Up_1(self, *args, **kwargs) -> None:

        self.pop = tk.Toplevel()
        self._Set_Smo_UI()

        return None

    def _Set_Smo_UI(self, *args, **kwargs) -> None:

        parent = self.pop

        parent.title('Autosubtraction')
        parent.geometry("400x400")

        self._Set_Smo_Numbers()
        self._Set_Smo_Folders()
        self._Set_Smo_Buttons()
        self._Set_Smo_Plots()

        return None

    def _Set_Smo_Buttons(self, *args, **kwargs) -> None:

        parent = self.pop
        reg = parent.register(self._Callback)

        button_use = Button(parent, text="Use", command=self._Smo_Use)
        button_use.place(height=30, width=184, x=8, y=16)

        button_cutoff = Button(parent, text="Cutoff", command=self._Smo_Cutoff)
        button_cutoff.place(height=30, width=88, x=208, y=16)

        var_q = StringVar()

        entry_q = Entry(parent, textvariable=var_q)
        entry_q.place(height=30, width=88, x=304, y=16)
        entry_q.config(validate="key", validatecommand=(reg, '%P'))
        entry_q.insert(0, f'{self.q_arr[10]:.5f}')

        self.smo_var_q = var_q

        return None

    def _Set_Smo_Plots(self, *args, **kwargs) -> None:

        parent = self.pop

        figure = Figure(figsize=(7, 8), dpi=64)

        plot = figure.add_subplot(1, 1, 1)
        plot.set_title("Raw")
        plot.plot(self.q_arr, self.I_arr)
        plot.axvline(x=self.q_arr[10], color='r')
        plot.set_xlabel(r'q ($\AA^{-1}$)')
        plot.set_ylabel('Scattering Intensity')
        plot.set_xscale('log')
        plot.set_yscale('log')

        canvas = FigureCanvasTkAgg(figure, parent)
        canvas.get_tk_widget().place(height=336, width=384, x=8, y=48)

        self.smo_figure = figure
        self.smo_plot = plot
        self.smo_canvas = canvas

        return None

    def _Set_Smo_Numbers(self, *args, **kwargs) -> None:

        self.x_0 = 0.0025
        self.is_smoothened = False
        self.qs_3 = self.q_arr
        self.Is_3 = self.I_arr

        return None

    def _Set_Smo_Folders(self, *args, **kwargs) -> None:

        cwd = os.getcwd()
        username = os.getlogin()
        current = datetime.now()
        current = current.strftime('%Y%m%d')

        base_path = os.path.join(cwd, 'Smoothening')

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        count = 0

        while True:
            temp = f'{username}_{current}_{count:02d}'
            if temp not in os.listdir(base_path):
                break
            else:
                count += 1

        working_dir = os.path.join(base_path, temp)
        smothened_dir = os.path.join(working_dir, 'Smoothened')
        img_dir = os.path.join(working_dir, 'Images')

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        if not os.path.exists(smothened_dir):
            os.makedirs(smothened_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        self.smo_base_path = base_path
        self.smo_working_dir = working_dir
        self.smo_smo_dir = smothened_dir
        self.smo_img_dir = img_dir

        return None

    def _Smo_Use(self) -> None:

        if self.is_smoothened:
            smo_dir = self.smo_smo_dir

            org_short = os.path.basename(self.file_path)
            org_name, _ = org_short.split('.')

            new_name = org_name + '_smooth' + '.csv'
            new_path = os.path.join(smo_dir, new_name)

            I_arr = self.I_arr = self.Is_3 / np.max(self.Is_3)
            q_arr = self.q_arr
            s_arr = np.sqrt(I_arr)

            data = np.hstack((
                q_arr.reshape(-1, 1),
                I_arr.reshape(-1, 1),
                s_arr.reshape(-1, 1)
            ))

            np.savetxt(new_path, data, delimiter=",")

            self.file_loaded = True
            self.folder_loaded = False
            self.origin = smo_dir
            self.file_path = new_path

            self._Draw_qI()
            self._Classify()
            self._Fit()

            self.pop.destroy()

        return None

    def _Smo_Update_Plot(self) -> None:

        if self.is_smoothened:
            self.smo_plot.clear()
            self.smo_plot.plot(self.q_arr, self.I_arr, label='Original')
            self.smo_plot.plot(self.qs_3, self.Is_3, label='Smoothened')
            self.smo_plot.axvline(x=self.q_low, color='r')
            self.smo_plot.set_title("Raw")
            self.smo_plot.set_xlabel(r'q ($\AA^{-1}$)')
            self.smo_plot.set_ylabel('Scattering Intensity')
            self.smo_plot.set_xscale('log')
            self.smo_plot.set_yscale('log')
            self.smo_plot.legend()
            self.smo_plot.grid()

            self.smo_canvas.draw()

        else:
            pass

        return None

    def _Smo_Update(self) -> None:

        self._Smo_Smoothen()
        self._Smo_Update_Plot()

        return None

    def _Smo_Smoothen(self) -> None:

        q_low = self.q_low

        xs = self.q_arr[self.q_arr >= q_low]
        ys = self.I_arr[self.q_arr >= q_low]

        x_0 = np.log10(self.q_arr[0])
        x_1 = np.log10(xs[0])
        x_2 = np.log10(xs[1])

        y_1 = np.log10(ys[0])
        y_2 = np.log10(ys[1])

        dy_1 = (y_2 - y_1) / (x_2 - x_1)

        a = dy_1 / (2 * x_1 * (x_1 - x_0))
        b = -2 * a * x_0
        c = y_1 - (x_1 - 2 * x_0) * dy_1 / (2 * (x_1 - x_0))

        x_ex = np.log10(self.q_arr[self.q_arr < q_low])
        y_ex = a * np.square(x_ex) + b * x_ex + c

        I_ex = np.power(10, y_ex)
        Is_3 = np.copy(self.I_arr)
        Is_3[self.q_arr < q_low] = I_ex

        self.qs_3 = self.q_arr
        self.Is_3 = Is_3
        self.ss_3 = np.sqrt(Is_3)
        self.is_smoothened = True

        return None

    def _Smo_Cutoff(self) -> None:

        q_low = float(self.smo_var_q.get())

        if q_low < self.q_arr[1]:
            q_low = self.q_arr[1]

        self.q_low = q_low
        self._Smo_Update()

        return None

    def _Callback(self, input_: str, *args, **kwargs) -> bool:

        """
        This checks if the input value is numerical or not.
        """

        try:
            if input_ == '':
                return True
            float(input_)
        except ValueError:
            return False
        return True

    def _Drop_Fit(self, input_: str, *args, **kwargs) -> None:

        """
        This allows the user to select the shape of the PCM.
        """

        self.shape = input_

        match input_:
            case 'Spheroid':
                self._class = 0
            case 'Cylinder':
                self._class = 1
            case _:
                pass

        if input_:
            if self.started:
                self.button_clear.configure(state=tk.NORMAL)
                self.button_export.configure(state=tk.NORMAL)
                self.button_simulate.configure(state=tk.NORMAL)
                self.button_visualize.configure(state=tk.NORMAL)
            else:
                self.button_clear.configure(state=tk.NORMAL)
                self.button_export.configure(state=tk.DISABLED)
                self.button_simulate.configure(state=tk.NORMAL)
                self.button_visualize.configure(state=tk.DISABLED)
        else:
            self.button_clear.configure(state=tk.DISABLED)
            self.button_export.configure(state=tk.DISABLED)
            self.button_simulate.configure(state=tk.DISABLED)
            self.button_visualize.configure(state=tk.DISABLED)

        return None

    def _Change_Mode(self, *args, **kwargs) -> None:

        if self.old_mode:

            self.new_mode = self.select_shapes.get()

            if self.new_mode == 'Spheroid':
                self.shape = 'Spheroid'
                self._class = 0
            elif self.new_mode == 'Cylinder':
                self.shape = 'Cylinder'
                self._class = 1
            else:
                pass

            self._Reconfigure()
            self.drop_methods.config(state=tk.NORMAL)
            self.select_shapes.set(self.shape)

            self._Fit()

            self.old_mode = self.select_shapes.get()

        return None

    def _ClearEntries(self, *args, **kwargs) -> None:

        self.entry_0_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_0_m.delete(0, tk.END)
        self.entry_0_m.config(state=tk.DISABLED, bg="Light grey")

        self.entry_0_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_0_s.delete(0, tk.END)
        self.entry_0_s.config(state=tk.DISABLED, bg="Light grey")

        self.entry_0_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_0_d.delete(0, tk.END)
        self.entry_0_d.config(state=tk.DISABLED, bg="Light grey")

        self.entry_1_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_1_m.delete(0, tk.END)
        self.entry_1_m.config(state=tk.DISABLED, bg="Light grey")

        self.entry_1_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_1_s.delete(0, tk.END)
        self.entry_1_s.config(state=tk.DISABLED, bg="Light grey")

        self.entry_1_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_1_d.delete(0, tk.END)
        self.entry_1_d.config(state=tk.DISABLED, bg="Light grey")

        self.entry_2_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_2_m.delete(0, tk.END)
        self.entry_2_m.config(state=tk.DISABLED, bg="Light grey")

        self.entry_2_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_2_s.delete(0, tk.END)
        self.entry_2_s.config(state=tk.DISABLED, bg="Light grey")

        self.entry_2_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_2_d.delete(0, tk.END)
        self.entry_2_d.config(state=tk.DISABLED, bg="Light grey")

        self.entry_0.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_0.delete(0, tk.END)
        self.entry_0.config(state=tk.DISABLED, bg="Light grey")

        self.entry_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_1.delete(0, tk.END)
        self.entry_1.config(state=tk.DISABLED, bg="Light grey")

        self.entry_2.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_2.delete(0, tk.END)
        self.entry_2.config(state=tk.DISABLED, bg="Light grey")

        self.entry_3.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_3.delete(0, tk.END)
        self.entry_3.config(state=tk.DISABLED, bg="Light grey")

        self.entry_4.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_4.delete(0, tk.END)
        self.entry_4.config(state=tk.DISABLED, bg="Light grey")

        self.entry_5.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_5.delete(0, tk.END)
        self.entry_5.config(state=tk.DISABLED, bg="Light grey")

        self.entry_6.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_6.delete(0, tk.END)
        self.entry_6.config(state=tk.DISABLED, bg="Light grey")

        self.entry_7.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_7.delete(0, tk.END)
        self.entry_7.config(state=tk.DISABLED, bg="Light grey")

        self.entry_8.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_8.delete(0, tk.END)
        self.entry_8.config(state=tk.DISABLED, bg="Light grey")

        self.entry_9.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_9.delete(0, tk.END)
        self.entry_9.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _ClearButtons(self, *args, **kwargs) -> None:

        self.button_0_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_0_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_0_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_0_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_1_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_1_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_1_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_1_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_2_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_2_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_2_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_2_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_3_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_3_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_3_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_3_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_4_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_4_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_4_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_4_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_5_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_5_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_5_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_5_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_6_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_6_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_6_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_6_N_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_7_P_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_7_N_L.config(state=tk.DISABLED, bg="Light grey")
        self.button_7_P_S.config(state=tk.DISABLED, bg="Light grey")
        self.button_7_N_S.config(state=tk.DISABLED, bg="Light grey")

        if not self.folder_loaded:
            self.button_forward.config(state=tk.DISABLED, bg="Light grey")
            self.button_backward.config(state=tk.DISABLED, bg="Light grey")
        else:
            pass

        return None

    def _ClearPlots(self, *args, **kwargs) -> None:

        self.plot_s.clear()
        self.plot_s.set_title("Loaded Sample")
        self.plot_s.set_xlabel(r'q ($\AA$)')
        self.plot_s.set_ylabel("Normalized Intensity")
        self.plot_s.set_xscale('log')
        self.plot_s.set_yscale('log')
        self.plot_s.grid()

        self.canvas_s.draw()

        if self._class == 0 or self._class == 1:
            self.plot_0.clear()
            self.plot_0.set_title("Radius Probability Function")
            self.plot_0.set_xlabel(r'Radius ($\AA$)')
            self.plot_0.set_ylabel('Probability Density')
            self.plot_0.grid()

            self.canvas_0.draw()

            self.plot_1.clear()
            self.plot_1.set_title("Aspect Ratio Probability Function")
            self.plot_1.set_xlabel(r'Aspect ratio (%)')
            self.plot_1.set_ylabel('Probability Density')
            self.plot_1.grid()

            self.canvas_1.draw()

            self.plot_2.clear()
            self.plot_2.set_title("PDI Probability Function")
            self.plot_2.set_xscale('log')
            self.plot_2.set_xlabel('PDI')
            self.plot_2.set_ylabel('Probability Density')
            self.plot_2.grid()

            self.plot_2.draw()

        else:
            self.plot_0.clear()
            self.plot_0.set_title("Radius Probability Function")
            self.plot_0.set_xlabel(r'Radius ($\AA$)')
            self.plot_0.set_ylabel('Probability Density')
            self.plot_0.grid()

            self.canvas_0.draw()

            self.plot_1.clear()
            self.plot_1.set_title("Length Probability Function")
            self.plot_1.set_xlabel(r'Length ($\AA$)')
            self.plot_1.set_ylabel('Probability Density')
            self.plot_1.grid()

            self.canvas_1.draw()

            self.plot_2.clear()
            self.plot_2.set_title("PDI Probability Function")
            self.plot_2.set_xscale('log')
            self.plot_2.set_xlabel('PDI')
            self.plot_2.set_ylabel('Probability Density')
            self.plot_2.grid()

            self.plot_2.draw()

        return None

    def _Clear(self, *args, **kwargs) -> None:
        self.file_loaded = False
        self.started = False
        self.fitted = False

        self._ToggleFeatures()
        self._ClearEntries()
        self._ClearButtons()

        self.label_count.config(text='N/A')
        return None

    def _Change_File(self, forward: bool, *args, **kwargs) -> None:

        """
        This function is for changing the file when a folder is loaded.

        1. Decide whether to move forward or backward in a folder.
        2. Get the file name.
        3. Get the file path.
        4. Prepare the file.
            a. Clear the plots, entries, and parameters.
            b. Get the new SAXS data.
            c. Draw the new SAXS data.
            d. Predict the shape.
            e. Predict the parameter values.
        5. Update the count.
        """

        # 1
        if forward:
            self.count = min(self.count + 1, self.max_count - 1)
        else:
            self.count = max(self.count - 1, 0)

        if self.count == self.max_count - 1:
            self.button_forward.config(state=tk.DISABLED, bg="Light grey")
            self.button_backward.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        elif self.count == 0:
            self.button_forward.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_backward.config(state=tk.DISABLED, bg="Light grey")
        else:
            self.button_forward.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_backward.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

        # 2
        filenameshort = self.dir_list[self.count]
        # 3
        filename = os.path.join(self.source_path, filenameshort)

        self.origin = self.source_path
        self.file_path = filename

        # 4
        self._PrepareFile()

        # 5
        self.label_count.config(text=f'{self.count + 1}/{self.max_count}')

        return None

    def _Forward(self, *args, **kwargs) -> None:
        self._Change_File(forward=True)
        return None

    def _Backward(self, *args, **kwargs) -> None:
        self._Change_File(forward=False)
        return None

    def _LoadFile(self, *args, **kwargs) -> None:
        self.update_status("Loading file...")
        """
        This function is for loading a single file.

        1. Get the file path.
        2. Get the source directory of the loaded file.
            a. This is to copy the file later.
        3. Prepare the file.
            a. Clear the plots, entries, and parameters.
            b. Get the new SAXS data.
            c. Draw the new SAXS data.
            d. Predict the shape.
            e. Predict the parameter values.
        """

        # 1
        root = self.parent

        root.filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select A File"
        )

        filename = root.filename

        if filename:
            # 2
            filenameshort = os.path.basename(filename)

            name_len = len(filenameshort)
            folder_name = filename[:-name_len]

            self.file_loaded = True
            self.folder_loaded = False
            self.origin = folder_name
            self.file_path = filename

            self._PrepareFile()

        self.update_status("File loaded successfully.")

        return None

    def _PrepareFile(self, *args, **kwargs) -> None:

        """
        This function is to prepare a file for viewing.

        1. Clear the plots, entries, and parameters.
        2. Get the new SAXS data.
        3. Draw the new SAXS data.
        4. Predict the shape.
        5. Predict the parameter values.
        """

        self._Clear()
        self.button_simulate.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.get_qI()
        self._Draw_qI()
        self._Classify()
        self._Fit()

        return None

    def _LoadFolder(self, *args, **kwargs) -> None:
        self.update_status("Loading folder...")
        """
        This function is for loading a folder only of SAXS data.

        1. Get the folder path.
        2. Clear plots, entries, and parameters.
        3. Start the count and move forward.

        """

        # 1
        root = self.parent

        root.path = filedialog.askdirectory(
            initialdir=os.getcwd(),
            title="Load Folder"
        )

        source_path = root.path

        if source_path:

            # 2
            self._Clear()

            self.button_file.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_clear.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_forward.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_backward.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

            self.source_path = source_path
            self.file_loaded = False
            self.folder_loaded = True
            self.dir_list = os.listdir(source_path)

            # 3
            self.count = -1
            self.max_count = len(self.dir_list)

            self._Forward()

        else:
            self.folder_loaded = False

        self.update_status("Folder loaded.")

        return None

    def _Export(self, *args, **kwargs) -> None:
        self.update_status("Exporting...")
        """
        This function is to store the parameters.

        1. Get the parameter values.
        2. Write them to the log.
        3. Copy the SAXS data to the new working directory.

        """

        # 1
        cwd = self.cwd
        origin = self.origin
        file_path = self.file_path

        shape = self.shape

        p_0 = self.p_0
        p_1 = self.p_1
        p_2 = self.p_2
        p_3 = self.p_3
        p_4 = self.p_4
        p_5 = self.p_5
        p_6 = self.p_6
        p_7 = self.p_7

        m_0 = self.m_0
        m_1 = self.m_1
        m_2 = self.m_2

        s_0 = self.s_0
        s_1 = self.s_1
        s_2 = self.s_2

        r_g_0 = self.r_g_0
        r_g_1 = self.r_g_1

        error = self.error

        comment = self.entry_comment.get()

        # 2
        log_path = self.log_path

        with open(log_path, 'a') as f:
            f.write(
                f'{cwd},{origin},{file_path},{shape},{p_0},{p_1},{p_2},{p_3},{p_4},{p_5},{p_6},{p_7},{m_0},{m_1},{m_2},{s_0},{s_1},{s_2},{error},{r_g_0},{r_g_1},{comment}\n')

        # 3
        target = os.path.join(self.working_dir, os.path.basename(self.file_path))
        shutil.copy(self.file_path, target)

        name = os.path.splitext(os.path.basename(file_path))[0]

        self.figure_s.savefig(os.path.join(self.working_dir, f"{name}_figure.png"))

        self.update_status("Data exported.")

        return None

    def _Change(self, param: str, change: str, *args, **kwargs) -> None:

        match param:

            case 0:

                match change:
                    case 0:
                        delta = 1.0
                    case 1:
                        delta = -1.0
                    case 2:
                        delta = 0.1
                    case 3:
                        delta = -0.1
                    case _:
                        pass

                self.p_0 += delta
                self.entry_0.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0.delete(0, tk.END)
                self.entry_0.insert(0, f'{self.p_0:.3f}')

            case 1:

                match change:
                    case 0:
                        delta = 0.1
                    case 1:
                        delta = -0.1
                    case 2:
                        delta = 0.01
                    case 3:
                        delta = -0.01
                    case _:
                        pass

                if self._class == 0:
                    self.p_1 += delta
                    self.entry_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                    self.entry_1.delete(0, tk.END)
                    self.entry_1.insert(0, f'{100 * self.p_1:.3f}')

                elif self._class == 1:
                    delta *= 10
                    self.p_1 += delta
                    self.entry_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                    self.entry_1.delete(0, tk.END)
                    self.entry_1.insert(0, f'{self.p_1:.3f}')
                else:
                    pass

            case 2:

                match change:
                    case 0:
                        delta = 0.1
                    case 1:
                        delta = -0.1
                    case 2:
                        delta = 0.01
                    case 3:
                        delta = -0.01
                    case _:
                        pass

                self.p_2 += delta
                self.entry_2.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2.delete(0, tk.END)
                self.entry_2.insert(0, f'{self.p_2:.3f}')

            case 3:

                match change:
                    case 0:
                        delta = 0.1
                    case 1:
                        delta = -0.1
                    case 2:
                        delta = 0.01
                    case 3:
                        delta = -0.01
                    case _:
                        pass

                self.p_3 += delta
                self.entry_3.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_3.delete(0, tk.END)
                self.entry_3.insert(0, f'{100 * self.p_3:.3f}')

            case 4:

                match change:
                    case 0:
                        delta = 0.001
                    case 1:
                        delta = -0.001
                    case 2:
                        delta = 0.000_1
                    case 3:
                        delta = -0.000_1
                    case _:
                        pass

                self.p_4 += delta
                self.entry_4.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_4.delete(0, tk.END)
                self.entry_4.insert(0, f'{1000 * self.p_4:.3f}')

            case 5:

                match change:
                    case 0:
                        delta = 1.0
                    case 1:
                        delta = -1.0
                    case 2:
                        delta = 0.1
                    case 3:
                        delta = -0.1
                    case _:
                        pass

                self.p_5 += delta
                self.entry_5.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_5.delete(0, tk.END)
                self.entry_5.insert(0, f'{self.p_5:.3f}')

            case 6:

                match change:
                    case 0:
                        delta = 0.1
                    case 1:
                        delta = -0.1
                    case 2:
                        delta = 0.01
                    case 3:
                        delta = -0.01
                    case _:
                        pass

                self.p_6 += delta
                self.entry_6.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_6.delete(0, tk.END)
                self.entry_6.insert(0, f'{self.p_6:.3f}')

            case 7:

                match change:
                    case 0:
                        delta = 0.1
                    case 1:
                        delta = -0.1
                    case 2:
                        delta = 0.01
                    case 3:
                        delta = -0.01
                    case _:
                        pass

                self.p_7 += delta
                self.entry_7.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')

            case _:
                pass

        return None

    def _Classify(self, *args, **kwargs) -> None:

        """
        1. Load the radius of gyration prediction and class prediction models.
        2. Prepare the SAXS data.
        3. Predict the radius of gyration.
        4. Change the axis from q to qr.
        5. Predict the class.
        6. Display the class chosen.
        7. Update the labels.
        """

        model_qr = self.model_qr
        model_cl = self.model_cl

        self._Prepare()
        X = self.X

        qr = model_qr.predict(X)[0]
        self.qr = np.power(10, 3 * qr)

        self.interpolate()
        Y = self.Y
        pred = model_cl.predict(Y)[0]

        likely = round(pred)
        pred = 100 * pred

        self.entry_class_0.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_class_0.delete(0, tk.END)
        self.entry_class_0.insert(0, f'{100 - pred}%')
        self.entry_class_0.config(state=tk.DISABLED, bg="Light grey")

        self.entry_class_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_class_1.delete(0, tk.END)
        self.entry_class_1.insert(0, f'{pred}%')
        self.entry_class_1.config(state=tk.DISABLED, bg="Light grey")

        match likely:
            case 0:
                self.shape = 'Spheroid'
                self._class = 0
                self.old_mode = 'Spheroid'
                self.new_mode = 'Spheroid'
            case 1:
                self.shape = 'Cylinder'
                self._class = 1
                self.old_mode = 'Cylinder'
                self.new_mode = 'Cylinder'

        self._Reconfigure()
        self.drop_methods.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.select_shapes.set(self.shape)

        return None

    def _Reconfigure(self, *args, **kwargs) -> None:

        self._EnableInputs()

        match self._class:
            case 0:
                self.label_1.config(text="Aspect Ratio")
                self.label_aux1.config(text="%")
                self.label_1_m.config(text="Mean (%):")
                self.label_1_s.config(text="STD (%):")

            case 1:
                self.label_1.config(text="Axial Length")
                self.label_aux1.config(text="Å")
                self.label_1_m.config(text="Mean (Å):")
                self.label_1_s.config(text="STD (Å):")

            case _:
                pass

        return None

    def _EnableInputs(self, *args, **kwargs) -> None:

        self.entry_0.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_2.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_3.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_4.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_5.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_6.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_7.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

        self.button_0_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_0_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_0_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_0_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_1_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_1_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_1_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_1_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_2_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_2_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_2_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_2_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_3_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_3_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_3_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_3_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_4_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_4_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_4_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_4_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_5_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_5_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_5_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_5_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_6_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_6_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_6_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_6_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_7_P_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_7_N_L.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_7_P_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.button_7_N_S.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

        return None

    def _ToggleFeatures(self, *args, **kwargs) -> None:

        if self.started:
            self.button_export.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_simulate.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_simulate.config(text='Simulate')
            self.button_visualize.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        else:
            self.button_export.config(state=tk.DISABLED, bg="Light grey")
            self.button_simulate.config(state=tk.DISABLED, bg="Light grey")
            self.button_simulate.config(text='Start')
            self.button_visualize.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _Fit(self, *args, **kwargs) -> None:

        """
        1. Check the class.
        2. Assign the models.
        3. Get the raw predictions from DL models.
        4. Translate the raw predictions into actual values.
        5. Display the prediction results.
        6. Update the GUI.
        """

        match self._class:
            case 0:
                self.model_0 = self.model_s_0
                self.model_1 = self.model_s_1
                self.model_2 = self.model_s_2
                self.model_3 = self.model_s_3
            case 1:
                self.model_0 = self.model_c_0
                self.model_1 = self.model_c_1
                self.model_2 = self.model_c_2
                self.model_3 = self.model_c_3
            case _:
                pass

        self._GetPrediction()
        self._Translate()
        self._DisplayParams()
        self._DisplayProbability()

        if self.fitted:
            self.drop_methods.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")

            self.button_export.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_visualize.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
            self.button_simulate.config(text='Simulate')

            self._EnableInputs()

        return None

    def _GetPrediction(self, *args, **kwargs) -> None:

        """
        1. Load the prepared SAXS data.
        2. Get Predictions.
        3. Store the raw predictions.
        """

        X = self.X

        pred_0 = self.model_0.predict(X)
        pred_1 = self.model_1.predict(X)
        pred_2 = self.model_2.predict(X)
        pred_3 = self.model_3.predict(X)

        m_0, s_0 = pred_0[0, 0], pred_0[0, 1]
        m_1, s_1 = pred_1[0, 0], pred_1[0, 1]
        m_2, s_2 = pred_2[0, 0], pred_2[0, 1]
        m_3 = pred_3[0, 0]

        self.m_0 = m_0
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3

        self.s_0 = s_0
        self.s_1 = s_1
        self.s_2 = s_2

        return None

    def _Translate(self, *args, **kwargs) -> None:

        """
        1. Check the class.
        2. Translate the raw predictions using the appropriate methods.
            p_0: core radius
            p_1: aspect ratio or length
            p_2: PDI
            p_3: fraction of scatterers in the core
            p_4: excess scattering length density ratio between corona and core
            p_5: shell thickness
            p_6: density factor for the core
                - sphere: about 2.0
                - cylinder: about 1.0
            p_7: density factor the the shell
                - Should be about 0.0
        """

        match self._class:
            case 0:
                self.p_0 = 256 * self.m_0
                self.p_1 = 2 * self.m_1
                self.p_2 = 10 ** (-4 * self.m_2) / 2
                self.p_3 = 0.75
                self.p_4 = 0.025
                self.p_5 = 2 * self.p_0
                self.p_6 = 2.0
                self.p_7 = 0.0

                self.STD_0 = 256 * self.s_0
                self.STD_1 = 2 * self.s_1
                self.STD_2 = self.s_2 / 2

                self.r_g_0 = 256 * self.m_3
                self.Guinier_fit()

            case 1:
                self.p_0 = 256 * self.m_0
                self.p_1 = 16 * self.m_1 * self.p_0
                self.p_2 = 10 ** (-4 * self.m_2) / 2
                self.p_3 = 0.75
                self.p_4 = 0.025
                self.p_5 = 2 * self.p_0
                self.p_6 = 1.0
                self.p_7 = 0.0

                self.STD_0 = 256 * self.s_0
                self.STD_1 = 16 * self.s_1 * self.p_0
                self.STD_2 = self.s_2 / 2

                self.r_g_0 = 1024 * self.m_3
                self.Guinier_fit()

                self.p_1 = np.sqrt(12 * (np.square(self.r_g_0) - np.square(self.p_0) / 2))

            case _:
                pass

        return None

    def _DisplayParams(self, *args, **kwargs) -> None:

        match self._class:

            case 0:

                self.entry_0.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0.delete(0, tk.END)
                self.entry_0.insert(0, f'{self.p_0:.3f}')

                self.entry_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1.delete(0, tk.END)
                self.entry_1.insert(0, f'{100 * self.p_1:.3f}')

                self.entry_2.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2.delete(0, tk.END)
                self.entry_2.insert(0, f'{self.p_2:.3f}')

                self.entry_3.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_3.delete(0, tk.END)
                self.entry_3.insert(0, f'{100 * self.p_3:.3f}')

                self.entry_4.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_4.delete(0, tk.END)
                self.entry_4.insert(0, f'{1000 * self.p_4:.3f}')

                self.entry_5.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_5.delete(0, tk.END)
                self.entry_5.insert(0, f'{self.p_5:.3f}')

                self.entry_6.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_6.delete(0, tk.END)
                self.entry_6.insert(0, f'{self.p_6:.3f}')

                self.entry_7.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')

                self.fitted = True

            case 1:

                self.entry_0.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0.delete(0, tk.END)
                self.entry_0.insert(0, f'{self.p_0:.3f}')

                self.entry_1.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1.delete(0, tk.END)
                self.entry_1.insert(0, f'{self.p_1:.3f}')

                self.entry_2.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2.delete(0, tk.END)
                self.entry_2.insert(0, f'{self.p_2:.3f}')

                self.entry_3.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_3.delete(0, tk.END)
                self.entry_3.insert(0, f'{100 * self.p_3:.3f}')

                self.entry_4.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_4.delete(0, tk.END)
                self.entry_4.insert(0, f'{1000 * self.p_4:.3f}')

                self.entry_5.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_5.delete(0, tk.END)
                self.entry_5.insert(0, f'{self.p_5:.3f}')

                self.entry_6.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_6.delete(0, tk.END)
                self.entry_6.insert(0, f'{self.p_6:.3f}')

                self.entry_7.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')

                self.fitted = True

            case _:
                pass

        self.entry_8.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_8.delete(0, tk.END)
        self.entry_8.insert(0, f'{self.r_g_0:.3f}')
        self.entry_8.config(state=tk.DISABLED, bg="Light grey")

        self.entry_9.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_9.delete(0, tk.END)
        self.entry_9.insert(0, f'{self.r_g_1:.3f}')
        self.entry_9.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _DisplayProbability(self, *args, **kwargs) -> None:

        match self._class:

            case 0:

                self.entry_0_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0_m.delete(0, tk.END)
                self.entry_0_m.insert(0, f'{self.p_0:.3f}')
                self.entry_0_m.config(state=tk.DISABLED, bg="Light grey")

                self.entry_0_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0_s.delete(0, tk.END)
                self.entry_0_s.insert(0, f'{self.STD_0:.3f}')
                self.entry_0_s.config(state=tk.DISABLED, bg="Light grey")

                self.entry_0_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0_d.delete(0, tk.END)
                self.entry_0_d.insert(0, '0')
                self.entry_0_d.config(state=tk.DISABLED, bg="Light grey")

                self.entry_1_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1_m.delete(0, tk.END)
                self.entry_1_m.insert(0, f'{100 * self.p_1:.3f}')
                self.entry_1_m.config(state=tk.DISABLED, bg="Light grey")

                self.entry_1_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1_s.delete(0, tk.END)
                self.entry_1_s.insert(0, f'{100 * self.STD_1:.3f}')
                self.entry_1_s.config(state=tk.DISABLED, bg="Light grey")

                self.entry_1_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1_d.delete(0, tk.END)
                self.entry_1_d.insert(0, '0')
                self.entry_1_d.config(state=tk.DISABLED, bg="Light grey")

                self.entry_2_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2_m.delete(0, tk.END)
                self.entry_2_m.insert(0, f'{self.p_2:.3f}')
                self.entry_2_m.config(state=tk.DISABLED, bg="Light grey")

                self.entry_2_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2_s.delete(0, tk.END)
                self.entry_2_s.insert(0, f'{self.STD_2:.3f}')
                self.entry_2_s.config(state=tk.DISABLED, bg="Light grey")

                self.entry_2_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2_d.delete(0, tk.END)
                self.entry_2_d.insert(0, '0')
                self.entry_2_d.config(state=tk.DISABLED, bg="Light grey")

            case 1:

                self.entry_0_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0_m.delete(0, tk.END)
                self.entry_0_m.insert(0, f'{self.p_0:.3f}')
                self.entry_0_m.config(state=tk.DISABLED, bg="Light grey")

                self.entry_0_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0_s.delete(0, tk.END)
                self.entry_0_s.insert(0, f'{self.STD_0:.3f}')
                self.entry_0_s.config(state=tk.DISABLED, bg="Light grey")

                self.entry_0_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_0_d.delete(0, tk.END)
                self.entry_0_d.insert(0, '0')
                self.entry_0_d.config(state=tk.DISABLED, bg="Light grey")

                self.entry_1_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1_m.delete(0, tk.END)
                self.entry_1_m.insert(0, f'{self.p_1:.3f}')
                self.entry_1_m.config(state=tk.DISABLED, bg="Light grey")

                self.entry_1_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1_s.delete(0, tk.END)
                self.entry_1_s.insert(0, f'{self.STD_1:.3f}')
                self.entry_1_s.config(state=tk.DISABLED, bg="Light grey")

                self.entry_1_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_1_d.delete(0, tk.END)
                self.entry_1_d.insert(0, '0')
                self.entry_1_d.config(state=tk.DISABLED, bg="Light grey")

                self.entry_2_m.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2_m.delete(0, tk.END)
                self.entry_2_m.insert(0, f'{self.p_2:.3f}')
                self.entry_2_m.config(state=tk.DISABLED, bg="Light grey")

                self.entry_2_s.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2_s.delete(0, tk.END)
                self.entry_2_s.insert(0, f'{self.STD_2:.3f}')
                self.entry_2_s.config(state=tk.DISABLED, bg="Light grey")

                self.entry_2_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
                self.entry_2_d.delete(0, tk.END)
                self.entry_2_d.insert(0, '0')
                self.entry_2_d.config(state=tk.DISABLED, bg="Light grey")

            case _:
                pass

        return None

    def _Simulate(self, *args, **kwargs) -> None:

        """
        1. Select the class.
        2. Create a micelle instance.
        3. Fetch the scattering simulation.
        4. Calculate the deviation from the real data.
        """

        match self._class:
            case 0:
                self.method = Disperse_Spheroid_Shell
                R = self.p_0
                epsilon = self.p_1
                PDI = self.p_2
                f_core = self.p_3
                rho_delta = self.p_4
                t = self.p_5
                p = self.p_6
                q = self.p_7
            case 1:
                self.method = Disperse_Cylinder_Shell
                R = self.p_0
                epsilon = self.p_1 / R
                PDI = self.p_2
                f_core = self.p_3
                rho_delta = self.p_4
                t = self.p_5
                p = self.p_6
                q = self.p_7
            case _:
                pass

        self.s = self.method(
            R=R,
            epsilon=epsilon,
            PDI=PDI,
            f_core=f_core,
            rho_delta=rho_delta,
            t=t,
            p=p,
            q=q
        )

        self.I_sim = self.s.Debye_scattering(q_arr=self.q_arr)

        self._Error()

        return None

    def _Error(self, *args, **kwargs) -> None:

        # Calculate the mean logarithmic squared error.
        # This focuses on the low-q region.

        error = np.mean(np.square(np.log(self.I_arr + 1) - np.log(self.I_sim + 1)))
        self.error = error

        self.Entry_MSLE.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.Entry_MSLE.delete(0, tk.END)
        self.Entry_MSLE.insert(0, f'{error * 1000:.3f}')
        self.Entry_MSLE.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _Probability(self, *args, **kwargs) -> None:

        # Create Gaussian distributions using the calculated parameters.

        match self._class:
            case 0:
                self._Deviance_0()
            case 1:
                self._Deviance_1()
            case _:
                pass

        temp = np.linspace(0, 2, 257)[:-1]

        prob_0 = np.exp(-np.square((temp - self.m_0) / (2 * self.s_0))) / self.s_0
        prob_1 = np.exp(-np.square((temp - self.m_1) / (2 * self.s_1))) / self.s_1
        prob_2 = np.exp(-np.square((temp - self.m_2) / (2 * self.s_2))) / self.s_2

        self.prob_0 = prob_0 / np.sqrt(2 * np.pi)
        self.prob_1 = prob_1 / np.sqrt(2 * np.pi)
        self.prob_2 = prob_2 / np.sqrt(2 * np.pi)

        self.entry_0_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_0_d.delete(0, tk.END)
        self.entry_0_d.insert(0, f'{self.dev_0:.3f}')
        self.entry_0_d.config(state=tk.DISABLED, bg="Light grey")

        self.entry_1_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_1_d.delete(0, tk.END)
        self.entry_1_d.insert(0, f'{self.dev_1:.3f}')
        self.entry_1_d.config(state=tk.DISABLED, bg="Light grey")

        self.entry_2_d.config(state=tk.NORMAL, background="#FFFFFF", foreground="black")
        self.entry_2_d.delete(0, tk.END)
        self.entry_2_d.insert(0, f'{self.dev_2:.3f}')
        self.entry_2_d.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _Deviance_0(self, *args, **kwargs) -> None:

        # How much the current parameter values deviate from the predicted mean.
        # For spheroids.
        # Should be zero initially.

        self.dev_0 = (self.p_0 / 256 - self.m_0) / self.s_0
        self.dev_1 = (self.p_1 / 2 - self.m_1) / self.s_1
        self.dev_2 = (-np.log10(self.p_2 * 2) / 4 - self.m_2) / self.s_2

        return None

    def _Deviance_1(self, *args, **kwargs) -> None:

        # How much the current parameter values deviate from the predicted mean.
        # For cylinders.
        # Should be zero initially.

        self.dev_0 = (self.p_0 / 256 - self.m_0) / self.s_0
        self.dev_1 = (self.p_1 / (16 * self.p_0) - self.m_1) / self.s_1
        self.dev_2 = (-np.log10(self.p_2 * 2) / 4 - self.m_2) / self.s_2

        return None

    def _Draw_qI(self, *args, **kwargs) -> None:

        filenameshort = os.path.basename(self.file_path)

        plot_s = self.plot_s
        canvas_s = self.canvas_s

        plot_s.clear()
        plot_s.plot(self.q_arr, self.I_arr, label='True')
        plot_s.set_title(filenameshort)
        plot_s.set_xlabel(r'q ($\AA$)')
        plot_s.set_ylabel("Normalized Intensity")
        plot_s.set_xscale('log')
        plot_s.set_yscale('log')
        plot_s.legend()
        plot_s.grid()

        canvas_s.draw()

        return None

    def _Draw_sim(self, *args, **kwargs) -> None:

        filenameshort = os.path.basename(self.file_path)

        plot_s = self.plot_s
        canvas_s = self.canvas_s

        plot_s.clear()
        plot_s.plot(self.q_arr, self.I_arr, label='True')
        plot_s.plot(self.q_arr, self.I_sim, label='Simulated')
        plot_s.set_title(filenameshort)
        plot_s.set_xlabel(r'q ($\AA$)')
        plot_s.set_ylabel("Normalized Intensity")
        plot_s.set_xscale('log')
        plot_s.set_yscale('log')
        plot_s.legend()
        plot_s.grid()

        canvas_s.draw()

        return None

    def _Draw_probability(self, *args, **kwargs) -> None:

        match self._class:
            case 0:
                self._Draw_probability_0()
            case 1:
                self._Draw_probability_1()
            case _:
                pass

        return None

    def _Draw_probability_0(self, *args, **kwargs) -> None:

        self._Probability()

        m_0 = self.m_0
        m_1 = self.m_1
        m_2 = self.m_2

        s_0 = self.s_0
        s_1 = self.s_1
        s_2 = self.s_2

        r_0 = 256 * (m_0 + 1.96 * s_0)
        l_0 = 256 * (m_0 - 1.96 * s_0)

        r_1 = 100 * (m_1 + 1.96 * s_1) * 2
        l_1 = 100 * (m_1 - 1.96 * s_1) * 2

        r_2 = (10 ** (-4 * (m_2 + 1.96 * s_2))) / 2
        l_2 = (10 ** (-4 * (m_2 - 1.96 * s_2))) / 2

        temp = np.linspace(0, 2, 257)[:-1]

        plot_0 = self.plot_0
        canvas_0 = self.canvas_0

        plot_1 = self.plot_1
        canvas_1 = self.canvas_1

        plot_2 = self.plot_2
        canvas_2 = self.canvas_2

        plot_0.clear()
        plot_0.plot(256 * temp, self.prob_0, color='blue')
        plot_0.axvline(self.p_0, color='red')
        plot_0.axvline(r_0, color='black', linestyle='dashed')
        plot_0.axvline(l_0, color='black', linestyle='dashed')
        plot_0.set_title("Radius Probability Function")
        plot_0.set_xlabel(r'Radius ($\AA$)')
        plot_0.set_ylabel(r'Probability Density')
        plot_0.grid()

        canvas_0.draw()

        plot_1.clear()
        plot_1.plot(100 * (temp * 2), self.prob_1, color='blue')
        plot_1.axvline(100 * self.p_1, color='red')
        plot_1.axvline(r_1, color='black', linestyle='dashed')
        plot_1.axvline(l_1, color='black', linestyle='dashed')
        plot_1.set_title("Aspect Ratio Probability Function")
        plot_1.set_xlabel(r'Aspect Ratio (%)')
        plot_1.set_ylabel(r'Probability Density')
        plot_1.grid()

        canvas_1.draw()

        plot_2.clear()
        plot_2.plot(np.power(10, -4 * temp) / 2, self.prob_2, color='blue')
        plot_2.axvline(self.p_2, color='red')
        plot_2.axvline(r_2, color='black', linestyle='dashed')
        plot_2.axvline(l_2, color='black', linestyle='dashed')
        plot_2.set_title("PDI Probability Function")
        plot_2.set_xlabel(r'PDI')
        plot_2.set_ylabel(r'Probability Density')
        plot_2.set_xscale('log')
        plot_2.grid()

        canvas_2.draw()

        return None

    def _Draw_probability_1(self, *args, **kwargs) -> None:

        self._Probability()

        m_0 = self.m_0
        m_1 = self.m_1
        m_2 = self.m_2

        s_0 = self.s_0
        s_1 = self.s_1
        s_2 = self.s_2

        r_0 = 256 * (m_0 + 1.96 * s_0)
        l_0 = 256 * (m_0 - 1.96 * s_0)

        r_1 = 16 * (m_1 + 1.96 * s_1) * self.p_0
        l_1 = 16 * (m_1 - 1.96 * s_1) * self.p_0

        r_2 = (m_2 + 1.96 * s_2) / 2
        l_2 = (m_2 - 1.96 * s_2) / 2

        temp = np.linspace(0, 2, 257)[:-1]

        plot_0 = self.plot_0
        canvas_0 = self.canvas_0

        plot_1 = self.plot_1
        canvas_1 = self.canvas_1

        plot_2 = self.plot_2
        canvas_2 = self.canvas_2

        plot_0.clear()
        plot_0.plot(256 * temp, self.prob_0, color='blue')
        plot_0.axvline(self.p_0, color='red')
        plot_0.axvline(r_0, color='black', linestyle='dashed')
        plot_0.axvline(l_0, color='black', linestyle='dashed')
        plot_0.set_title("Radius Probability Function")
        plot_0.set_xlabel(r'Radius ($\AA$)')
        plot_0.set_ylabel(r'Probability Density')
        plot_0.grid()

        canvas_0.draw()

        plot_1.clear()
        plot_1.plot(16 * temp * self.p_0, self.prob_1, color='blue')
        plot_1.axvline(self.p_1, color='red')
        plot_1.axvline(r_1, color='black', linestyle='dashed')
        plot_1.axvline(l_1, color='black', linestyle='dashed')
        plot_1.set_title("Aspect Ratio Probability Function")
        plot_1.set_xlabel(r'Aspect Ratio (%)')
        plot_1.set_ylabel(r'Probability Density')
        plot_1.grid()

        canvas_1.draw()

        plot_2.clear()
        plot_2.plot(np.power(10, -4 * temp) / 2, self.prob_2, color='blue')
        plot_2.axvline(self.p_2, color='red')
        plot_2.axvline(r_2, color='black', linestyle='dashed')
        plot_2.axvline(l_2, color='black', linestyle='dashed')
        plot_2.set_title("PDI Probability Function")
        plot_2.set_xlabel(r'PDI')
        plot_2.set_ylabel(r'Probability Density')
        plot_2.set_xscale('log')
        plot_2.grid()

        canvas_2.draw()

        return None

    def _Simulate_as(self, *args, **kwargs) -> None:
        self.update_status("Starting simulation...")

        if not self.started:

            if self.fitted:
                self.started = True
                self._ToggleFeatures()
                self._Simulate()
                self._Draw_sim()
                self._Draw_probability()

        match self._class:
            case 0:
                self._Simulate_as_0()
            case 1:
                self._Simulate_as_1()
            case _:
                pass

        self.update_status("Simulation completed.")

        return None

    def _Simulate_as_0(self, *args, **kwargs) -> None:

        p_0 = self.entry_0.get()
        p_0 = float(p_0)
        self.p_0 = p_0

        p_1 = self.entry_1.get()
        p_1 = float(p_1) / 100
        self.p_1 = p_1

        p_2 = self.entry_2.get()
        p_2 = float(p_2)
        self.p_2 = p_2

        p_3 = self.entry_3.get()
        p_3 = float(p_3) / 100
        self.p_3 = p_3

        p_4 = self.entry_4.get()
        p_4 = float(p_4) / 1000
        self.p_4 = p_4

        p_5 = self.entry_5.get()
        p_5 = float(p_5)
        self.p_5 = p_5

        p_6 = self.entry_6.get()
        p_6 = float(p_6)
        self.p_6 = p_6

        p_7 = self.entry_7.get()
        p_7 = float(p_7)
        self.p_7 = p_7

        self._Simulate()
        self._Draw_sim()
        self._Draw_probability()

        return None

    def _Simulate_as_1(self, *args, **kwargs) -> None:

        p_0 = self.entry_0.get()
        p_0 = float(p_0)
        self.p_0 = p_0

        p_1 = self.entry_1.get()
        p_1 = float(p_1)
        self.p_1 = p_1

        p_2 = self.entry_2.get()
        p_2 = float(p_2)
        self.p_2 = p_2

        p_3 = self.entry_3.get()
        p_3 = float(p_3) / 100
        self.p_3 = p_3

        p_4 = self.entry_4.get()
        p_4 = float(p_4) / 1000
        self.p_4 = p_4

        p_5 = self.entry_5.get()
        p_5 = float(p_5)
        self.p_5 = p_5

        p_6 = self.entry_6.get()
        p_6 = float(p_6)
        self.p_6 = p_6

        p_7 = self.entry_7.get()
        p_7 = float(p_7)
        self.p_7 = p_7

        self._Simulate()
        self._Draw_sim()
        self._Draw_probability()

        return None

    def _Visualize(self, *args, **kwargs) -> None:
        self.update_status("Visualizing results...")
        n = 4096
        s = self.s

        scatterers, _ = s.generate_scatterers(n=n)
        scatterers *= self.p_0

        xs, ys, zs = scatterers[:, 0], scatterers[:, 1], scatterers[:, 2]

        root = self.parent

        top = tk.Toplevel(root)
        top.geometry("720x720")
        top.title("Simulation Visualization")

        fig = Figure(figsize=(5, 4), dpi=64)

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()

        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(xs, ys, zs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', adjustable='box')

        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.update_status("Visualization complete.")

        return None

    def _Autosubtraction(self, *args, **kwargs) -> None:
        self._Pop_Up_0()
        return None

    def get_qI(self, *args, **kwargs) -> None:

        file_path = self.file_path

        filenameshort = os.path.basename(file_path)
        end = filenameshort[-3:]

        if 'button' in kwargs:
            _button = kwargs['button']
            _button.configure(text=filenameshort)

        temp = list()

        with open(file_path, 'r') as f:

            while f.readline().lstrip()[0] not in "0123456789":
                continue

            while True:
                line = f.readline().lstrip()

                if line:
                    if end == 'csv':
                        num = len(line.split(','))

                        if num == 2:
                            q, I = line.split(',')
                        else:
                            q, I, _ = line.split(',')
                    else:
                        num = len(line.split())

                        if num == 2:
                            q, I = line.split()
                        else:
                            q, I, _ = line.split()

                    temp.append((float(q), float(I)))

                else:
                    break

        temp = np.array(temp)

        qs = temp[:, 0]
        Is = temp[:, 1]

        Is[Is <= 0] = np.min(Is[Is > 0])

        Is = np.interp(self.q_arr, qs, Is)
        Is /= np.max(Is)

        self.I_arr = Is

        return None

    def _Prepare(self, *args, **kwargs) -> None:

        X = self.I_arr

        X[X <= 0] = np.min(X[X >= 0])
        X = 1 + np.log10(X) / 2
        X = np.tanh(X)
        X = X[np.newaxis, :, np.newaxis]

        self.X = X

        return None

    def _Help(self, *args, **kwargs) -> None:
        text = """
        Important parameters:
        ----------------------
        These parameters significantly affect the scattering simulation:

        - Radius: The radius of the core of the micelle
        - Aspect Ratio: The height-to-width ratio for spheroidal micelles
        - Length: The length of cylindrical micelles
        - PDI (Polydispersity Index): The polydispersity of the micelle
        - Core Fraction: The fraction of the micelle electrons located in the core
        - Scattering Fraction: The percentage of the corona excess scattering length
         density relative to that of the core

        Arbitrary parameters:
        ----------------------
        These parameters do not significantly affect the scattering simulation:

        - Corona Length: The length of the corona of the micelles
        - Core Density: A value between 0 and 2 for spheroidal micelles and 0 and 1
         for cylindrical micelles. As values approach 2 (or 1), the density becomes 
         uniform.
        - Corona Density: Similar to core density, but refers to the corona of the 
          micelles.
        - mMSLE: milli-mean-squared-logarithmic error; an arbitrary error rate that 
          gauges the fit of the simulation, favoring the Guinier region.
        """

        pop = tk.Toplevel()

        pop.title('Help')
        pop.geometry("480x512")

        label_help = Label(pop, text=text)
        label_help.place(x=0, y=0)

        return None

    def interpolate(self, *args, **kwargs) -> None:

        X_int = np.linspace(np.log10(0.64), np.log10(10.0), 64)
        X_ref = self.q_log_arr + np.log10(self.qr)
        Y_ref = np.log10(self.I_arr)

        self.Y = np.interp(x=X_int, xp=X_ref, fp=Y_ref)[np.newaxis, :]

        return None

    def Guinier_fit(self, *args, **kwargs) -> None:

        r_g = self.r_g_0

        q_arr = self.q_arr
        I_arr = self.I_arr

        qr = q_arr * r_g

        match self._class:
            case 0:
                cut = 1.3
            case 1:
                cut = 1.0
            case _:
                pass

        q_new = q_arr[qr <= cut]
        I_new = I_arr[qr <= cut]

        if q_new.size < 8:
            q_new = q_arr[:8]
            I_new = I_arr[:8]

        X = np.square(q_new)
        Y = np.log(I_new)

        a, _ = np.polyfit(x=X, y=Y, deg=1)
        R_g = sqrt(-3 * a)

        self.r_g_1 = R_g

        return None


def main(*args, **kwargs) -> int:
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

    return 0


if __name__ == '__main__':
    main()
