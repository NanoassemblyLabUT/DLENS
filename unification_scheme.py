import os
import shutil

import numpy as np
import tkinter as tk
import tensorflow as tf
import pickle as pk

from datetime import datetime

from tkinter import Button, StringVar, filedialog, Entry, Label, OptionMenu

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from math import sqrt

from Debye import Disperse_Spheroid_Shell, Disperse_Cylinder_Shell

"""
The Debye module is a file that works alongside this file.
"""


class MainApplication(tk.Frame):
    
    
    def __init__(self, parent, *args, **kwargs):
        
        self.parent = parent
        
        self._Setting()
        self._Layout()
        self._LoadModels()
        
        return None
    
    
    def _Setting(self, *args, **kwargs) -> None:
        
        # This is the standardized q values used for the inference.
        q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
        q_arr = np.power(10, q_log_arr - 2*np.log10(2))
        
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
            f.write('CWD,From,File,Param 1,Param 2,Param 3,Param 4,Param 5,Param 6,Param 7,mu 1,mu 2,sigma 1,sigma 2,Error,R_g,R_g,Comment\n')
        
        self.log_file = log_file
        self.base_path = base_path
        self.working_dir = working_dir
        self.log_path = log_path
        
        self.file_loaded = False
        self.fitted = False
        self.started = False
        self.folder_loaded = False
        
        self.shape = None
        self._class = None
        self.speed = 1
        self.rho = 0.001/self.speed
        self.count = -1
                
        return None
    
    
    def _LayOperation(self, *args, **kwargs) -> None:
        
        parent = self.parent
        dy = self.dy
        
        button_folder = Button(parent, text="Load Folder", command=self._LoadFolder)
        button_folder.place(height=2*dy, width=128, x=16, y=1*dy)
        
        button_file = Button(parent, text="Load File", command=self._LoadFile)
        button_file.place(height=2*dy, width=128, x=16, y=3*dy)
        
        shapes = ["Spheroid", "Cylinder"]
        select_shapes = StringVar()
        select_shapes.set("Fitting Algorithm")
        
        drop_methods = OptionMenu(parent, select_shapes, *shapes, command=self._Drop_Fit)
        drop_methods.config(width=20)
        drop_methods.place(height=30, width=128, x=16, y=5*dy)
        drop_methods.config(state=tk.DISABLED)
        
        button_clear = Button(parent, text="Clear", command=self._Clear)
        button_clear.place(height=30, width=128, x=16, y=7*dy)
        button_clear.config(state=tk.DISABLED)
        
        button_export = Button(parent, text="Export", command=self._Export)
        button_export.place(height=30, width=128, x=16, y=9*dy)
        button_export.config(state=tk.DISABLED)
        
        button_simulate = Button(parent, text="Start", command=self._Simulate_as)
        button_simulate.place(height=30, width=128, x=16, y=11*dy)
        button_simulate.config(state=tk.DISABLED)

        button_visualize = Button(parent, text="Visualize", command=self._Visualize)
        button_visualize.place(height=30, width=128, x=16, y=13*dy)
        button_visualize.config(state=tk.DISABLED)
        
        label_speed = Label(parent, text="Speed")
        label_speed.place(height=30, width=64, x=16, y=15*dy)
        
        speeds = ["1", "2", "3", "4"]
        select_speeds = StringVar()
        select_speeds.set("1")
        
        drop_speeds = OptionMenu(parent, select_speeds, *speeds, command=self._Drop_Speed)
        drop_speeds.config(width=20)
        drop_speeds.place(height=30, width=64, x=80, y=15*dy)
        drop_speeds.config(state=tk.DISABLED)
        
        label_count = Label(parent, text="N/A")
        label_count.place(height=30, width=64, x=16, y=17*dy)
        
        button_backward = Button(parent, text='<', command=self._Backward)
        button_backward.place(height=30, width=30, x=80, y=17*dy)
        button_backward.config(state=tk.DISABLED)
        
        button_forward = Button(parent, text='>', command=self._Forward)
        button_forward.place(height=30, width=30, x=112, y=17*dy)
        button_forward.config(state=tk.DISABLED)
        
        label_MSLE = Label(parent, text=u"\u03bcMSLE:")
        label_MSLE.place(height=30, width=128, x=144, y=17*dy)
        
        var_MSLE = StringVar()
                
        Entry_MSLE = Entry(parent, textvariable=var_MSLE)
        Entry_MSLE.place(height=30, width=96, x=272, y=17*dy)
        Entry_MSLE.config(state=tk.DISABLED)
        
        label_comment = Label(parent, text="Comment:")
        label_comment.place(height=30, width=128, x=16, y=19*dy)
        
        var_comment = StringVar()
        
        entry_comment = Entry(parent, textvariable=var_comment)
        entry_comment.place(height=30, width=192, x=144, y=19*dy)
        entry_comment.config(state=tk.NORMAL)
        
        label__class0 = Label(parent, text="Spheroid:")
        label__class0.place(height=30, width=64, x=336, y=19*dy)
        
        var__class0 = StringVar()
        
        entry__class0 = Entry(parent, textvariable=var__class0)
        entry__class0.place(height=30, width=64, x=400, y=19*dy)
        entry__class0.config(state=tk.DISABLED)
        
        label__class1 = Label(parent, text="Cylinder:")
        label__class1.place(height=30, width=64, x=464, y=19*dy)
        
        var__class1 = StringVar()
        
        entry__class1 = Entry(parent, textvariable=var__class1)
        entry__class1.place(height=30, width=64, x=528, y=19*dy)
        entry__class1.config(state=tk.DISABLED)
        
        self.drop_methods = drop_methods
        self.drop_speeds = drop_speeds
        
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
        
        self.entry__class0 = entry__class0
        self.entry__class1 = entry__class1
        
        return None
    
    
    def _LayParameters(self, *args, **kwargs) -> None:
        
        parent = self.parent
        dy = self.dy
        
        reg = parent.register(self._Callback)
        
        label_0 = Label(parent, text="Radius:")
        label_0.place(height=30, width=128, x=144, y=1*dy)

        var_0 = StringVar()

        entry_0 = Entry(parent, textvariable=var_0)
        entry_0.place(height=30, width=96, x=272, y=1*dy)
        entry_0.config(state=tk.DISABLED)
        entry_0.config(validate="key", validatecommand=(reg, '%P'))
        
        label_aux0 = Label(parent, text='Å')
        label_aux0.place(height=30, width=16, x=368, y=1*dy)
        
        label_1 = Label(parent, text="Aspect Ratio:")
        label_1.place(height=30, width=128, x=144, y=3*dy)
        
        var_1 = StringVar()
        
        entry_1 = Entry(parent, textvariable=var_1)
        entry_1.place(height=30, width=96, x=272, y=3*dy)
        entry_1.config(state=tk.DISABLED)
        entry_1.config(validate="key", validatecommand=(reg, '%P'))
        
        label_aux1 = Label(parent, text='%', justify="left")
        label_aux1.place(height=30, width=16, x=368, y=3*dy)
        
        label_2 = Label(parent, text="PDI:")
        label_2.place(height=30, width=128, x=144, y=5*dy)
        
        var_2 = StringVar()
        
        entry_2 = Entry(parent, textvariable=var_2)
        entry_2.place(height=30, width=96, x=272, y=5*dy)
        entry_2.config(state=tk.DISABLED)
        entry_2.config(validate="key", validatecommand=(reg, '%P'))
        
        label_3 = Label(parent, text="Core Fraction:")
        label_3.place(height=30, width=128, x=144, y=7*dy)
        
        var_3 = StringVar()
        
        entry_3 = Entry(parent, textvariable=var_3)
        entry_3.place(height=30, width=96, x=272, y=7*dy)
        entry_3.config(state=tk.DISABLED)
        entry_3.config(validate="key", validatecommand=(reg, '%P'))
        
        label_aux2 = Label(parent, text='%')
        label_aux2.place(height=30, width=16, x=368, y=7*dy)
        
        label_4 = Label(parent, text="Scattering Fraction:")
        label_4.place(height=30, width=128, x=144, y=9*dy)
        
        var_4 = StringVar()
        
        entry_4 = Entry(parent, textvariable=var_4)
        entry_4.place(height=30, width=96, x=272, y=9*dy)
        entry_4.config(state=tk.DISABLED)
        entry_4.config(validate="key", validatecommand=(reg, '%P'))
        
        label_aux3 = Label(parent, text='‰')
        label_aux3.place(height=30, width=16, x=368, y=9*dy)
        
        label_5 = Label(parent, text="Corona Length:")
        label_5.place(height=30, width=128, x=144, y=11*dy)
        
        var_5 = StringVar()
        
        entry_5 = Entry(parent, textvariable=var_5)
        entry_5.place(height=30, width=96, x=272, y=11*dy)
        entry_5.config(state=tk.DISABLED)
        entry_5.config(validate="key", validatecommand=(reg, '%P'))
        
        label_aux4 = Label(parent, text='Å')
        label_aux4.place(height=30, width=16, x=368, y=11*dy)
        
        label_6 = Label(parent, text="Core Density:")
        label_6.place(height=30, width=128, x=144, y=13*dy)
        
        var_6 = StringVar()
        
        entry_6 = Entry(parent, textvariable=var_6)
        entry_6.place(height=30, width=96, x=272, y=13*dy)
        entry_6.config(state=tk.DISABLED)
        entry_6.config(validate="key", validatecommand=(reg, '%P'))
        
        label_7 = Label(parent, text="Corona Density:")
        label_7.place(height=30, width=128, x=144, y=15*dy)
        
        var_7 = StringVar()
        
        entry_7 = Entry(parent, textvariable=var_7)
        entry_7.place(height=30, width=96, x=272, y=15*dy)
        entry_7.config(state=tk.DISABLED)
        entry_7.config(validate="key", validatecommand=(reg, '%P'))
        
        label_8 = Label(parent, text=r'R_g (ML)')
        label_8.place(height=30, width=96, x=512, y=1*dy)
        
        var_8 = StringVar()
        
        entry_8 = Entry(parent, textvariable=var_8)
        entry_8.place(height=30, width=96, x=512, y=3*dy)
        entry_8.config(state=tk.DISABLED)
        entry_8.config(validate="key", validatecommand=(reg, '%P'))
        
        label_9 = Label(parent, text=r'R_g (GN)')
        label_9.place(height=30, width=96, x=512, y=5*dy)
        
        var_9 = StringVar()
        
        entry_9 = Entry(parent, textvariable=var_9)
        entry_9.place(height=30, width=96, x=512, y=7*dy)
        entry_9.config(state=tk.DISABLED)
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
        button_0_P_L.place(height=30, width=30, x=384, y=1*dy)
        button_0_P_L.config(state=tk.DISABLED)
        
        button_0_N_L = Button(parent, text='-', command=lambda: self._Change(0, 1))
        button_0_N_L.place(height=30, width=30, x=416, y=1*dy)
        button_0_N_L.config(state=tk.DISABLED)
        
        button_0_P_S = Button(parent, text='+', command=lambda: self._Change(0, 2))
        button_0_P_S.place(height=20, width=20, x=448, y=1*dy + 5)
        button_0_P_S.config(state=tk.DISABLED)
        
        button_0_N_S = Button(parent, text='-', command=lambda: self._Change(0, 3))
        button_0_N_S.place(height=20, width=20, x=480, y=1*dy + 5)
        button_0_N_S.config(state=tk.DISABLED)
        
        button_1_P_L = Button(parent, text='+', command=lambda: self._Change(1, 0))
        button_1_P_L.place(height=30, width=30, x=384, y=3*dy)
        button_1_P_L.config(state=tk.DISABLED)
        
        button_1_N_L = Button(parent, text='-', command=lambda: self._Change(1, 1))
        button_1_N_L.place(height=30, width=30, x=416, y=3*dy)
        button_1_N_L.config(state=tk.DISABLED)
        
        button_1_P_S = Button(parent, text='+', command=lambda: self._Change(1, 2))
        button_1_P_S.place(height=20, width=20, x=448, y=3*dy + 5)
        button_1_P_S.config(state=tk.DISABLED)
        
        button_1_N_S = Button(parent, text='-', command=lambda: self._Change(1, 3))
        button_1_N_S.place(height=20, width=20, x=480, y=3*dy + 5)
        button_1_N_S.config(state=tk.DISABLED)
        
        button_2_P_L = Button(parent, text='+', command=lambda: self._Change(2, 0))
        button_2_P_L.place(height=30, width=30, x=384, y=5*dy)
        button_2_P_L.config(state=tk.DISABLED)
        
        button_2_N_L = Button(parent, text='-', command=lambda: self._Change(2, 1))
        button_2_N_L.place(height=30, width=30, x=416, y=5*dy)
        button_2_N_L.config(state=tk.DISABLED)
        
        button_2_P_S = Button(parent, text='+', command=lambda: self._Change(2, 2))
        button_2_P_S.place(height=20, width=20, x=448, y=5*dy + 5)
        button_2_P_S.config(state=tk.DISABLED)
        
        button_2_N_S = Button(parent, text='-', command=lambda: self._Change(2, 3))
        button_2_N_S.place(height=20, width=20, x=480, y=5*dy + 5)
        button_2_N_S.config(state=tk.DISABLED)
        
        button_3_P_L = Button(parent, text='+', command=lambda: self._Change(3, 0))
        button_3_P_L.place(height=30, width=30, x=384, y=7*dy)
        button_3_P_L.config(state=tk.DISABLED)
        
        button_3_N_L = Button(parent, text='-', command=lambda: self._Change(3, 1))
        button_3_N_L.place(height=30, width=30, x=416, y=7*dy)
        button_3_N_L.config(state=tk.DISABLED)
        
        button_3_P_S = Button(parent, text='+', command=lambda: self._Change(3, 2))
        button_3_P_S.place(height=20, width=20, x=448, y=7*dy + 5)
        button_3_P_S.config(state=tk.DISABLED)
        
        button_3_N_S = Button(parent, text='-', command=lambda: self._Change(3, 3))
        button_3_N_S.place(height=20, width=20, x=480, y=7*dy + 5)
        button_3_N_S.config(state=tk.DISABLED)
        
        button_4_P_L = Button(parent, text='+', command=lambda: self._Change(4, 0))
        button_4_P_L.place(height=30, width=30, x=384, y=9*dy)
        button_4_P_L.config(state=tk.DISABLED)
        
        button_4_N_L = Button(parent, text='-', command=lambda: self._Change(4, 1))
        button_4_N_L.place(height=30, width=30, x=416, y=9*dy)
        button_4_N_L.config(state=tk.DISABLED)
        
        button_4_P_S = Button(parent, text='+', command=lambda: self._Change(4, 2))
        button_4_P_S.place(height=20, width=20, x=448, y=9*dy + 5)
        button_4_P_S.config(state=tk.DISABLED)
        
        button_4_N_S = Button(parent, text='-', command=lambda: self._Change(4, 3))
        button_4_N_S.place(height=20, width=20, x=480, y=9*dy + 5)
        button_4_N_S.config(state=tk.DISABLED)
        
        button_5_P_L = Button(parent, text='+', command=lambda: self._Change(5, 0))
        button_5_P_L.place(height=30, width=30, x=384, y=11*dy)
        button_5_P_L.config(state=tk.DISABLED)
        
        button_5_N_L = Button(parent, text='-', command=lambda: self._Change(5, 1))
        button_5_N_L.place(height=30, width=30, x=416, y=11*dy)
        button_5_N_L.config(state=tk.DISABLED)
        
        button_5_P_S = Button(parent, text='+', command=lambda: self._Change(5, 2))
        button_5_P_S.place(height=20, width=20, x=448, y=11*dy + 5)
        button_5_P_S.config(state=tk.DISABLED)
        
        button_5_N_S = Button(parent, text='-', command=lambda: self._Change(5, 3))
        button_5_N_S.place(height=20, width=20, x=480, y=11*dy + 5)
        button_5_N_S.config(state=tk.DISABLED)
        
        button_6_P_L = Button(parent, text='+', command=lambda: self._Change(6, 0))
        button_6_P_L.place(height=30, width=30, x=384, y=13*dy)
        button_6_P_L.config(state=tk.DISABLED)
        
        button_6_N_L = Button(parent, text='-', command=lambda: self._Change(6, 1))
        button_6_N_L.place(height=30, width=30, x=416, y=13*dy)
        button_6_N_L.config(state=tk.DISABLED)
        
        button_6_P_S = Button(parent, text='+', command=lambda: self._Change(6, 2))
        button_6_P_S.place(height=20, width=20, x=448, y=13*dy + 5)
        button_6_P_S.config(state=tk.DISABLED)
        
        button_6_N_S = Button(parent, text='-', command=lambda: self._Change(6, 3))
        button_6_N_S.place(height=20, width=20, x=480, y=13*dy + 5)
        button_6_N_S.config(state=tk.DISABLED)
        
        button_7_P_L = Button(parent, text='+', command=lambda: self._Change(7, 0))
        button_7_P_L.place(height=30, width=30, x=384, y=15*dy)
        button_7_P_L.config(state=tk.DISABLED)
        
        button_7_N_L = Button(parent, text='-', command=lambda: self._Change(7, 1))
        button_7_N_L.place(height=30, width=30, x=416, y=15*dy)
        button_7_N_L.config(state=tk.DISABLED)
        
        button_7_P_S = Button(parent, text='+', command=lambda: self._Change(7, 2))
        button_7_P_S.place(height=20, width=20, x=448, y=15*dy + 5)
        button_7_P_S.config(state=tk.DISABLED)
        
        button_7_N_S = Button(parent, text='-', command=lambda: self._Change(7, 3))
        button_7_N_S.place(height=20, width=20, x=480, y=15*dy + 5)
        button_7_N_S.config(state=tk.DISABLED)
        
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
        canvas_s.get_tk_widget().place(height=336, width=560, x=640, y=0*dy)
        
        figure_0 = Figure(figsize=(4, 3), dpi=64)
        
        plot_0 = figure_0.add_subplot(1, 1, 1)
        plot_0.set_title("Radius Probability Distribution")
        plot_0.set_xlabel(r'Radius ($\AA$)')
        plot_0.set_ylabel('Probability Density')
                
        canvas_0 = FigureCanvasTkAgg(figure_0, parent)
        canvas_0.get_tk_widget().place(height=300, width=400, x=16, y=22*dy)
        
        label_0_m = Label(parent, text="Mean (Å):")
        label_0_m.place(height=30, width=64, x=16, y=41*dy)
        
        var_0_m = StringVar()
        
        entry_0_m = Entry(parent, textvariable=var_0_m)
        entry_0_m.place(height=30, width=64, x=80, y=41*dy)
        entry_0_m.config(state=tk.DISABLED)
        entry_0_m.config(validate="key", validatecommand=(reg, '%P'))
        
        label_0_s = Label(parent, text="STD (Å):")
        label_0_s.place(height=30, width=64, x=144, y=41*dy)
        
        var_0_s = StringVar()
        
        entry_0_s = Entry(parent, textvariable=var_0_s)
        entry_0_s.place(height=30, width=64, x=208, y=41*dy)
        entry_0_s.config(state=tk.DISABLED)
        entry_0_s.config(validate="key", validatecommand=(reg, '%P'))
        
        label_0_d = Label(parent, text="Deviation:", justify="left")
        label_0_d.place(height=30, width=64, x=272, y=41*dy)
        
        var_0_d = StringVar()
        
        entry_0_d = Entry(parent, textvariable=var_0_d)
        entry_0_d.place(height=30, width=64, x=336, y=41*dy)
        entry_0_d.config(state=tk.DISABLED)
        entry_0_d.config(validate="key", validatecommand=(reg, '%P'))
        
        figure_1 = Figure(figsize=(4, 3), dpi=64)
        
        plot_1 = figure_1.add_subplot(1, 1, 1)
        plot_1.set_title("Aspect Ratio Probability Distribution")
        plot_1.set_xlabel(r'Aspect ratio (%)')
        plot_1.set_ylabel('Probability Density')
                
        canvas_1 = FigureCanvasTkAgg(figure_1, parent)
        canvas_1.get_tk_widget().place(height=300, width=400, x=432, y=22*dy)
        
        label_1_m = Label(parent, text="Mean (%):")
        label_1_m.place(height=30, width=64, x=432, y=41*dy)
        
        var_1_m = StringVar()
        
        entry_1_m = Entry(parent, textvariable=var_1_m)
        entry_1_m.place(height=30, width=64, x=496, y=41*dy)
        entry_1_m.config(state=tk.DISABLED)
        entry_1_m.config(validate="key", validatecommand=(reg, '%P'))
        
        label_1_s = Label(parent, text="STD (%):")
        label_1_s.place(height=30, width=64, x=560, y=41*dy)
        
        var_1_s = StringVar()
        
        entry_1_s = Entry(parent, textvariable=var_1_s)
        entry_1_s.place(height=30, width=64, x=624, y=41*dy)
        entry_1_s.config(state=tk.DISABLED)
        entry_1_s.config(validate="key", validatecommand=(reg, '%P'))
        
        label_1_d = Label(parent, text="Deviation:")
        label_1_d.place(height=30, width=64, x=688, y=41*dy)
        
        var_1_d = StringVar()
        
        entry_1_d = Entry(parent, textvariable=var_1_d)
        entry_1_d.place(height=30, width=64, x=752, y=41*dy)
        entry_1_d.config(state=tk.DISABLED)
        entry_1_d.config(validate="key", validatecommand=(reg, '%P'))
        
        figure_2 = Figure(figsize=(4, 3), dpi=64)
        
        plot_2 = figure_2.add_subplot(1, 1, 1)
        plot_2.set_title("p_2 Probability Distribution")
        plot_2.set_xlabel(r'p_2')
        plot_2.set_xscale('log')
        plot_2.set_ylabel('Probability Density')
                
        canvas_2 = FigureCanvasTkAgg(figure_2, parent)
        canvas_2.get_tk_widget().place(height=300, width=400, x=864, y=22*dy)
        
        label_2_m = Label(parent, text="Mean:")
        label_2_m.place(height=30, width=64, x=864, y=41*dy)
        
        var_2_m = StringVar()
        
        entry_2_m = Entry(parent, textvariable=var_2_m)
        entry_2_m.place(height=30, width=64, x=928, y=41*dy)
        entry_2_m.config(state=tk.DISABLED)
        entry_2_m.config(validate="key", validatecommand=(reg, '%P'))
        
        label_2_s = Label(parent, text="STD:")
        label_2_s.place(height=30, width=64, x=992, y=41*dy)
        
        var_2_s = StringVar()
        
        entry_2_s = Entry(parent, textvariable=var_2_s)
        entry_2_s.place(height=30, width=64, x=1056, y=41*dy)
        entry_2_s.config(state=tk.DISABLED)
        entry_2_s.config(validate="key", validatecommand=(reg, '%P'))
        
        label_2_d = Label(parent, text="Deviation:")
        label_2_d.place(height=30, width=64, x=1120, y=41*dy)
        
        var_2_d = StringVar()
        
        entry_2_d = Entry(parent, textvariable=var_2_d)
        entry_2_d.place(height=30, width=64, x=1184, y=41*dy)
        entry_2_d.config(state=tk.DISABLED)
        entry_2_d.config(validate="key", validatecommand=(reg, '%P'))
        
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
        
        cwd = self.cwd
        
        model_dir = os.path.join(cwd, 'Models')
        
        name_s_0 = "2024_11_18_sphere_CPNN_Radius_0.keras"
        name_s_1 = "2024_11_18_sphere_CPNN_AspectRatio_0.keras"
        name_s_2 = '2024_12_10_sphere_CPNN_PDI_0.keras'
        name_c_0 = "2024_11_18_cylinder_CPNN_Radius_0.keras"
        name_c_1 = "2024_11_18_cylinder_CPNN_AspectRatio_0.keras"
        name_c_2 = '2024_12_10_cylinder_CPNN_PDI_0.keras'
        name_qr = "2024_11_18_SCNN_qr_0.keras"
        name_cl = "2024_11_17_SVM_C_0.pkl"
        
        path_s_0 = os.path.join(model_dir, name_s_0)
        path_s_1 = os.path.join(model_dir, name_s_1)
        path_s_2 = os.path.join(model_dir, name_s_2)
        path_c_0 = os.path.join(model_dir, name_c_0)
        path_c_1 = os.path.join(model_dir, name_c_1)
        path_c_2 = os.path.join(model_dir, name_c_2)
        path_qr = os.path.join(model_dir, name_qr)
        path_cl = os.path.join(model_dir, name_cl)
        
        model_s_0 = tf.keras.models.load_model(path_s_0, compile=False)
        model_s_1 = tf.keras.models.load_model(path_s_1, compile=False)
        model_s_2 = tf.keras.models.load_model(path_s_2, compile=False)
        model_c_0 = tf.keras.models.load_model(path_c_0, compile=False)
        model_c_1 = tf.keras.models.load_model(path_c_1, compile=False)
        model_c_2 = tf.keras.models.load_model(path_c_2, compile=False)
        model_qr = tf.keras.models.load_model(path_qr, compile=False)
        
        with open(path_cl, 'rb') as f:
            model_cl = pk.load(f)
        
        self.model_s_0 = model_s_0
        self.model_s_1 = model_s_1
        self.model_s_2 = model_s_2
        self.model_c_0 = model_c_0
        self.model_c_1 = model_c_1
        self.model_c_2 = model_c_2
        self.model_qr = model_qr
        self.model_cl = model_cl
        
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
    
    
    def _Drop_Speed(self, input_: str, *args, **kwargs) -> None:
        self.speed = int(input_) + 1
        return None
    
    
    def _ClearEntries(self, *args, **kwargs) -> None:
        
        self.entry_0_m.config(state=tk.NORMAL)
        self.entry_0_m.delete(0, tk.END)
        self.entry_0_m.config(state=tk.DISABLED)
        
        self.entry_0_s.config(state=tk.NORMAL)
        self.entry_0_s.delete(0, tk.END)
        self.entry_0_s.config(state=tk.DISABLED)
        
        self.entry_0_d.config(state=tk.NORMAL)
        self.entry_0_d.delete(0, tk.END)
        self.entry_0_d.config(state=tk.DISABLED)
        
        self.entry_1_m.config(state=tk.NORMAL)
        self.entry_1_m.delete(0, tk.END)
        self.entry_1_m.config(state=tk.DISABLED)
        
        self.entry_1_s.config(state=tk.NORMAL)
        self.entry_1_s.delete(0, tk.END)
        self.entry_1_s.config(state=tk.DISABLED)
        
        self.entry_1_d.config(state=tk.NORMAL)
        self.entry_1_d.delete(0, tk.END)
        self.entry_1_d.config(state=tk.DISABLED)
        
        self.entry_2_m.config(state=tk.NORMAL)
        self.entry_2_m.delete(0, tk.END)
        self.entry_2_m.config(state=tk.DISABLED)
        
        self.entry_2_s.config(state=tk.NORMAL)
        self.entry_2_s.delete(0, tk.END)
        self.entry_2_s.config(state=tk.DISABLED)
        
        self.entry_2_d.config(state=tk.NORMAL)
        self.entry_2_d.delete(0, tk.END)
        self.entry_2_d.config(state=tk.DISABLED)
        
        self.entry_0.config(state=tk.NORMAL)
        self.entry_0.delete(0, tk.END)
        self.entry_0.config(state=tk.DISABLED)
        
        self.entry_1.config(state=tk.NORMAL)
        self.entry_1.delete(0, tk.END)
        self.entry_1.config(state=tk.DISABLED)
        
        self.entry_2.config(state=tk.NORMAL)
        self.entry_2.delete(0, tk.END)
        self.entry_2.config(state=tk.DISABLED)
        
        self.entry_3.config(state=tk.NORMAL)
        self.entry_3.delete(0, tk.END)
        self.entry_3.config(state=tk.DISABLED)
        
        self.entry_4.config(state=tk.NORMAL)
        self.entry_4.delete(0, tk.END)
        self.entry_4.config(state=tk.DISABLED)
        
        self.entry_5.config(state=tk.NORMAL)
        self.entry_5.delete(0, tk.END)
        self.entry_5.config(state=tk.DISABLED)
        
        self.entry_6.config(state=tk.NORMAL)
        self.entry_6.delete(0, tk.END)
        self.entry_6.config(state=tk.DISABLED)
        
        self.entry_7.config(state=tk.NORMAL)
        self.entry_7.delete(0, tk.END)
        self.entry_7.config(state=tk.DISABLED)
        
        self.entry_8.config(state=tk.NORMAL)
        self.entry_8.delete(0, tk.END)
        self.entry_8.config(state=tk.DISABLED)
        
        self.entry_9.config(state=tk.NORMAL)
        self.entry_9.delete(0, tk.END)
        self.entry_9.config(state=tk.DISABLED)
        
        return None
    
    
    def _ClearButtons(self, *args, **kwargs) -> None:
        
        self.button_0_P_L.config(state=tk.DISABLED)
        self.button_0_N_L.config(state=tk.DISABLED)
        self.button_0_P_S.config(state=tk.DISABLED)
        self.button_0_N_S.config(state=tk.DISABLED)
        self.button_1_P_L.config(state=tk.DISABLED)
        self.button_1_N_L.config(state=tk.DISABLED)
        self.button_1_P_S.config(state=tk.DISABLED)
        self.button_1_N_S.config(state=tk.DISABLED)
        self.button_2_P_L.config(state=tk.DISABLED)
        self.button_2_N_L.config(state=tk.DISABLED)
        self.button_2_P_S.config(state=tk.DISABLED)
        self.button_2_N_S.config(state=tk.DISABLED)
        self.button_3_P_L.config(state=tk.DISABLED)
        self.button_3_N_L.config(state=tk.DISABLED)
        self.button_3_P_S.config(state=tk.DISABLED)
        self.button_3_N_S.config(state=tk.DISABLED)
        self.button_4_P_L.config(state=tk.DISABLED)
        self.button_4_N_L.config(state=tk.DISABLED)
        self.button_4_P_S.config(state=tk.DISABLED)
        self.button_4_N_S.config(state=tk.DISABLED)
        self.button_5_P_L.config(state=tk.DISABLED)
        self.button_5_N_L.config(state=tk.DISABLED)
        self.button_5_P_S.config(state=tk.DISABLED)
        self.button_5_N_S.config(state=tk.DISABLED)
        self.button_6_P_L.config(state=tk.DISABLED)
        self.button_6_N_L.config(state=tk.DISABLED)
        self.button_6_P_S.config(state=tk.DISABLED)
        self.button_6_N_S.config(state=tk.DISABLED)
        self.button_7_P_L.config(state=tk.DISABLED)
        self.button_7_N_L.config(state=tk.DISABLED)
        self.button_7_P_S.config(state=tk.DISABLED)
        self.button_7_N_S.config(state=tk.DISABLED)
        
        if not self.folder_loaded:
            self.button_forward.config(state=tk.DISABLED)
            self.button_backward.config(state=tk.DISABLED)
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
            self.plot_2.set_title("p_2 Probability Function")
            self.plot_2.set_xscale('log')
            self.plot_2.set_xlabel('p_2')
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
            self.plot_2.set_title("p_2 Probability Function")
            self.plot_2.set_xscale('log')
            self.plot_2.set_xlabel('p_2')
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
            self.button_forward.config(state=tk.DISABLED)
        elif self.count == 0:
            self.button_backward.config(state=tk.DISABLED)
        else:
            self.button_forward.config(state=tk.NORMAL)
            self.button_backward.config(state=tk.NORMAL)
        
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
        self.button_simulate.config(state=tk.NORMAL)
        self.get_qI()
        self._Draw_qI()
        self._Classify()
        self._Fit()
        
        return None
    
    
    def _LoadFolder(self, *args, **kwargs) -> None:
        
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
            
            self.button_file.config(state=tk.NORMAL)
            self.button_clear.config(state=tk.NORMAL)
            self.button_forward.config(state=tk.NORMAL)
            self.button_backward.config(state=tk.NORMAL)
            
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
        
        return None
    
    
    def _Export(self, *args, **kwargs) -> None:
        
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
        
        with open(log_path, 'w') as f:
            f.write(f'{cwd},{origin},{file_path},{p_0},{p_1},{p_2},{p_3},{p_4},{p_5},{p_6},{p_7},{m_0},{m_1},{m_2},{s_0},{s_1},{s_2},{error},{r_g_0},{r_g_1},{comment}\n')
        
        # 3
        target = os.path.join(self.working_dir, os.path.basename(self.file_path))
        shutil.copy(self.file_path, target)
        
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
                self.entry_0.config(state=tk.NORMAL)
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
                    self.entry_1.config(state=tk.NORMAL)
                    self.entry_1.delete(0, tk.END)
                    self.entry_1.insert(0, f'{100*self.p_1:.3f}')
                
                elif self._class == 1:
                    delta *= 10
                    self.p_1 += delta
                    self.entry_1.config(state=tk.NORMAL)
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
                self.entry_2.config(state=tk.NORMAL)
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
                self.entry_3.config(state=tk.NORMAL)
                self.entry_3.delete(0, tk.END)
                self.entry_3.insert(0, f'{100*self.p_3:.3f}')
            
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
                self.entry_4.config(state=tk.NORMAL)
                self.entry_4.delete(0, tk.END)
                self.entry_4.insert(0, f'{1000*self.p_4:.3f}')
            
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
                self.entry_5.config(state=tk.NORMAL)
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
                self.entry_6.config(state=tk.NORMAL)
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
                self.entry_7.config(state=tk.NORMAL)
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')

            case _:
                pass
        
        return None
    
    
    def _Classify(self, *args, **kwargs) -> None:
        
        model_qr = self.model_qr
        model_cl = self.model_cl
        
        self._Prepare()
        X = self.X
                
        self.qr = model_qr.predict(X)[0]*256
        
        self.interpolate()
        Y = self.Y
        pred = model_cl.predict(Y)[0]
        
        likely = round(pred)
        pred = 100*pred
                
        self.entry__class0.config(state=tk.NORMAL)
        self.entry__class0.delete(0, tk.END)
        self.entry__class0.insert(0, f'{100 - pred}%')
        self.entry__class0.config(state=tk.DISABLED)
        
        self.entry__class1.config(state=tk.NORMAL)
        self.entry__class1.delete(0, tk.END)
        self.entry__class1.insert(0, f'{pred}%')
        self.entry__class1.config(state=tk.DISABLED)
        
        match likely:
            case 0:
                self.shape = 'Spheroid'
                self._class = 0
            case 1:
                self.shape = 'Cylinder'
                self._class = 1
        
        self._Reconfigure()
        self.drop_methods.config(state=tk.NORMAL)
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
        
        self.entry_0.config(state=tk.NORMAL)
        self.entry_1.config(state=tk.NORMAL)
        self.entry_2.config(state=tk.NORMAL)
        self.entry_3.config(state=tk.NORMAL)
        self.entry_4.config(state=tk.NORMAL)
        self.entry_5.config(state=tk.NORMAL)
        self.entry_6.config(state=tk.NORMAL)
        self.entry_7.config(state=tk.NORMAL)
        
        self.button_0_P_L.config(state=tk.NORMAL)
        self.button_0_N_L.config(state=tk.NORMAL)
        self.button_0_P_S.config(state=tk.NORMAL)
        self.button_0_N_S.config(state=tk.NORMAL)
        self.button_1_P_L.config(state=tk.NORMAL)
        self.button_1_N_L.config(state=tk.NORMAL)
        self.button_1_P_S.config(state=tk.NORMAL)
        self.button_1_N_S.config(state=tk.NORMAL)
        self.button_2_P_L.config(state=tk.NORMAL)
        self.button_2_N_L.config(state=tk.NORMAL)
        self.button_2_P_S.config(state=tk.NORMAL)
        self.button_2_N_S.config(state=tk.NORMAL)
        self.button_3_P_L.config(state=tk.NORMAL)
        self.button_3_N_L.config(state=tk.NORMAL)
        self.button_3_P_S.config(state=tk.NORMAL)
        self.button_3_N_S.config(state=tk.NORMAL)
        self.button_4_P_L.config(state=tk.NORMAL)
        self.button_4_N_L.config(state=tk.NORMAL)
        self.button_4_P_S.config(state=tk.NORMAL)
        self.button_4_N_S.config(state=tk.NORMAL)
        self.button_5_P_L.config(state=tk.NORMAL)
        self.button_5_N_L.config(state=tk.NORMAL)
        self.button_5_P_S.config(state=tk.NORMAL)
        self.button_5_N_S.config(state=tk.NORMAL)
        self.button_6_P_L.config(state=tk.NORMAL)
        self.button_6_N_L.config(state=tk.NORMAL)
        self.button_6_P_S.config(state=tk.NORMAL)
        self.button_6_N_S.config(state=tk.NORMAL)
        self.button_7_P_L.config(state=tk.NORMAL)
        self.button_7_N_L.config(state=tk.NORMAL)
        self.button_7_P_S.config(state=tk.NORMAL)
        self.button_7_N_S.config(state=tk.NORMAL)
                
        return None
    
    
    def _ToggleFeatures(self, *args, **kwargs) -> None:
        
        if self.started:
            self.button_export.config(state=tk.NORMAL)
            self.button_simulate.config(state=tk.NORMAL)
            self.button_simulate.config(text='Simulate')
            self.button_visualize.config(state=tk.NORMAL)
        else:
            self.button_export.config(state=tk.DISABLED)
            self.button_simulate.config(state=tk.DISABLED)
            self.button_simulate.config(text='Start')
            self.button_visualize.config(state=tk.DISABLED)
        
        return None
    
    
    def _Fit(self, *args, **kwargs) -> None:
                        
        match self._class:
            case 0:
                self.model_0 = self.model_s_0
                self.model_1= self.model_s_1
                self.model_2 = self.model_s_2
            case 1:
                self.model_0 = self.model_c_0
                self.model_1= self.model_c_1
                self.model_2 = self.model_c_2
            case _:
                pass
        
        self._GetPrediction()
        self._Translate()
        self._DisplayParams()
        self._DisplayProbability()
        
        if self.fitted:
            
            self.drop_methods.config(state=tk.NORMAL)
            self.drop_speeds.config(state=tk.NORMAL)

            self.button_export.config(state=tk.NORMAL)
            self.button_visualize.config(state=tk.NORMAL)
            self.button_simulate.config(text='Simulate')
            
            self._EnableInputs()
            
        return None
    
    
    def _GetPrediction(self, *args, **kwargs) -> None:
        
        X = self.X
        
        pred_0 = self.model_0.predict(X)
        pred_1 = self.model_1.predict(X)
        pred_2 = self.model_2.predict(X)
        
        m_0, s_0 = pred_0[0, 0], pred_0[0, 1]
        m_1, s_1 = pred_1[0, 0], pred_1[0, 1]
        m_2, s_2 = pred_2[0, 0], pred_2[0, 1]        
                        
        self.m_0 = m_0
        self.m_1 = m_1
        self.m_2 = m_2
        self.s_0 = s_0
        self.s_1 = s_1
        self.s_2 = s_2
        
        return None

    
    def _Translate(self, *args, **kwargs) -> None:
                
        match self._class:
            case 0:
                self.p_0 = 256*self.m_0
                self.p_1 = 2*self.m_1
                self.p_2 = 10**(-self.m_2*4)/2
                self.p_3 = 0.75
                self.p_4 = 0.025
                self.p_5 = 2*self.p_0
                self.p_6 = 2.0
                self.p_7 = 0.0
                
                self.STD_0 = 256*self.s_0
                self.STD_1 = 2*self.s_1
                self.STD_2 = self.s_2/2
                
                a = -0.931_372_402_843_532_4
                b = 0.554_837_200_488_491_8
                c = 0.007_910_143_709_179_732
                
                corr = a*self.p_2**2 + b*self.p_2 + c + 1
                
                a = b = self.p_0
                c = self.p_1*self.p_0
                self.r_g_0 = sqrt((a**2 + b**2 + c**2)/5)/corr
                self.Guinier_fit()
                
            case 1:
                self.p_0 = 256*self.m_0
                self.p_1 = 16*self.m_1*self.p_0
                self.p_2 = 10**(-self.m_2*4)/2
                self.p_3 = 0.75
                self.p_4 = 0.025
                self.p_5 = 2*self.p_0
                self.p_6 = 1.0
                self.p_7 = 0.0
                
                self.STD_0 = 256*self.s_0
                self.STD_1 = 16*self.s_1*self.p_0
                self.STD_2 = self.s_2/2
                
                self.r_g_0 = sqrt(self.p_0**2/2 + self.p_1**2/12)
                self.Guinier_fit()
                
            case _:
                pass
        
        return None
    

    def _DisplayParams(self, *args, **kwargs) -> None:
        
        match self._class:
            
            case 0:
    
                self.entry_0.config(state=tk.NORMAL)
                self.entry_0.delete(0, tk.END)
                self.entry_0.insert(0, f'{self.p_0:.3f}')
                
                self.entry_1.config(state=tk.NORMAL)
                self.entry_1.delete(0, tk.END)
                self.entry_1.insert(0, f'{100*self.p_1:.3f}')
                
                self.entry_2.config(state=tk.NORMAL)
                self.entry_2.delete(0, tk.END)
                self.entry_2.insert(0, f'{self.p_2:.3f}')
                
                self.entry_3.config(state=tk.NORMAL)
                self.entry_3.delete(0, tk.END)
                self.entry_3.insert(0, f'{100*self.p_3:.3f}')
                
                self.entry_4.config(state=tk.NORMAL)
                self.entry_4.delete(0, tk.END)
                self.entry_4.insert(0, f'{1000*self.p_4:.3f}')
                
                self.entry_5.config(state=tk.NORMAL)
                self.entry_5.delete(0, tk.END)
                self.entry_5.insert(0, f'{self.p_5:.3f}')
                
                self.entry_6.config(state=tk.NORMAL)
                self.entry_6.delete(0, tk.END)
                self.entry_6.insert(0, f'{self.p_6:.3f}')
                
                self.entry_7.config(state=tk.NORMAL)
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')
                
                self.fitted = True
                
            case 1:
                
                self.entry_0.config(state=tk.NORMAL)
                self.entry_0.delete(0, tk.END)
                self.entry_0.insert(0, f'{self.p_0:.3f}')
                
                self.entry_1.config(state=tk.NORMAL)
                self.entry_1.delete(0, tk.END)
                self.entry_1.insert(0, f'{self.p_1:.3f}')
                
                self.entry_2.config(state=tk.NORMAL)
                self.entry_2.delete(0, tk.END)
                self.entry_2.insert(0, f'{self.p_2:.3f}')
                
                self.entry_3.config(state=tk.NORMAL)
                self.entry_3.delete(0, tk.END)
                self.entry_3.insert(0, f'{100*self.p_3:.3f}')
                
                self.entry_4.config(state=tk.NORMAL)
                self.entry_4.delete(0, tk.END)
                self.entry_4.insert(0, f'{1000*self.p_4:.3f}')
                
                self.entry_5.config(state=tk.NORMAL)
                self.entry_5.delete(0, tk.END)
                self.entry_5.insert(0, f'{self.p_5:.3f}')
                
                self.entry_6.config(state=tk.NORMAL)
                self.entry_6.delete(0, tk.END)
                self.entry_6.insert(0, f'{self.p_6:.3f}')
                
                self.entry_7.config(state=tk.NORMAL)
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')
                
                self.fitted = True
                
            case _:
                pass
        
        self.entry_8.config(state=tk.NORMAL)
        self.entry_8.delete(0, tk.END)
        self.entry_8.insert(0, f'{self.r_g_0:.3f}')
        self.entry_8.config(state=tk.DISABLED)

        self.entry_9.config(state=tk.NORMAL)
        self.entry_9.delete(0, tk.END)
        self.entry_9.insert(0, f'{self.r_g_1:.3f}')
        self.entry_9.config(state=tk.DISABLED)

        return None

    
    def _DisplayProbability(self, *args, **kwargs) -> None:
        
        match self._class:
            
            case 0:
    
                self.entry_0_m.config(state=tk.NORMAL)
                self.entry_0_m.delete(0, tk.END)
                self.entry_0_m.insert(0, f'{self.p_0:.3f}')
                self.entry_0_m.config(state=tk.DISABLED)
                
                self.entry_0_s.config(state=tk.NORMAL)
                self.entry_0_s.delete(0, tk.END)
                self.entry_0_s.insert(0, f'{self.STD_0:.3f}')
                self.entry_0_s.config(state=tk.DISABLED)
                
                self.entry_0_d.config(state=tk.NORMAL)
                self.entry_0_d.delete(0, tk.END)
                self.entry_0_d.insert(0, '0')
                self.entry_0_d.config(state=tk.DISABLED)
                
                self.entry_1_m.config(state=tk.NORMAL)
                self.entry_1_m.delete(0, tk.END)
                self.entry_1_m.insert(0, f'{100*self.p_1:.3f}')
                self.entry_1_m.config(state=tk.DISABLED)
                
                self.entry_1_s.config(state=tk.NORMAL)
                self.entry_1_s.delete(0, tk.END)
                self.entry_1_s.insert(0, f'{100*self.STD_1:.3f}')
                self.entry_1_s.config(state=tk.DISABLED)
                
                self.entry_1_d.config(state=tk.NORMAL)
                self.entry_1_d.delete(0, tk.END)
                self.entry_1_d.insert(0, '0')
                self.entry_1_d.config(state=tk.DISABLED)
                
                self.entry_2_m.config(state=tk.NORMAL)
                self.entry_2_m.delete(0, tk.END)
                self.entry_2_m.insert(0, f'{self.p_2:.3f}')
                self.entry_2_m.config(state=tk.DISABLED)
                
                self.entry_2_s.config(state=tk.NORMAL)
                self.entry_2_s.delete(0, tk.END)
                self.entry_2_s.insert(0, f'{self.STD_2:.3f}')
                self.entry_2_s.config(state=tk.DISABLED)
                
                self.entry_2_d.config(state=tk.NORMAL)
                self.entry_2_d.delete(0, tk.END)
                self.entry_2_d.insert(0, '0')
                self.entry_2_d.config(state=tk.DISABLED)
                
            case 1:
                
                self.entry_0_m.config(state=tk.NORMAL)
                self.entry_0_m.delete(0, tk.END)
                self.entry_0_m.insert(0, f'{self.p_0:.3f}')
                self.entry_0_m.config(state=tk.DISABLED)
                
                self.entry_0_s.config(state=tk.NORMAL)
                self.entry_0_s.delete(0, tk.END)
                self.entry_0_s.insert(0, f'{self.STD_0:.3f}')
                self.entry_0_s.config(state=tk.DISABLED)
                
                self.entry_0_d.config(state=tk.NORMAL)
                self.entry_0_d.delete(0, tk.END)
                self.entry_0_d.insert(0, '0')
                self.entry_0_d.config(state=tk.DISABLED)
                
                self.entry_1_m.config(state=tk.NORMAL)
                self.entry_1_m.delete(0, tk.END)
                self.entry_1_m.insert(0, f'{self.p_1:.3f}')
                self.entry_1_m.config(state=tk.DISABLED)
                
                self.entry_1_s.config(state=tk.NORMAL)
                self.entry_1_s.delete(0, tk.END)
                self.entry_1_s.insert(0, f'{self.STD_1:.3f}')
                self.entry_1_s.config(state=tk.DISABLED)
                
                self.entry_1_d.config(state=tk.NORMAL)
                self.entry_1_d.delete(0, tk.END)
                self.entry_1_d.insert(0, '0')
                self.entry_1_d.config(state=tk.DISABLED)
                
                self.entry_2_m.config(state=tk.NORMAL)
                self.entry_2_m.delete(0, tk.END)
                self.entry_2_m.insert(0, f'{self.p_2:.3f}')
                self.entry_2_m.config(state=tk.DISABLED)
                
                self.entry_2_s.config(state=tk.NORMAL)
                self.entry_2_s.delete(0, tk.END)
                self.entry_2_s.insert(0, f'{self.STD_2:.3f}')
                self.entry_2_s.config(state=tk.DISABLED)
                
                self.entry_2_d.config(state=tk.NORMAL)
                self.entry_2_d.delete(0, tk.END)
                self.entry_2_d.insert(0, '0')
                self.entry_2_d.config(state=tk.DISABLED)
                
            case _:
                pass
        
        return None
        
    
    def _Simulate(self, *args, **kwargs) -> None:
                
        self.rho = 0.001/(1 + self.speed)
        
        match self._class:
            case 0:
                self.method = Disperse_Spheroid_Shell
            case 1:
                self.method = Disperse_Cylinder_Shell
            case _:
                pass
        
        self.s = self.method(
            self.p_0, 
            self.p_1, 
            self.p_2, 
            self.p_3, 
            self.p_4, 
            self.p_5, 
            self.p_6, 
            self.p_7
        )
        self.I_sim = self.s.Debye_Scattering(q_arr=self.q_arr)
        
        self._Error()
                
        return None
    
    
    def _Error(self, *args, **kwargs) -> None:
        
        error = np.mean(np.square(np.log(self.I_arr + 1) - np.log(self.I_sim + 1)))
        self.error = error
        
        self.Entry_MSLE.config(state=tk.NORMAL)
        self.Entry_MSLE.delete(0, tk.END)
        self.Entry_MSLE.insert(0, f'{error*1000:.3f}')
        self.Entry_MSLE.config(state=tk.DISABLED)
        
        return None
    
    
    def _Probability(self, *args, **kwargs) -> None:
        
        match self._class:
            case 0:
                self._Deviance_0()
            case 1:
                self._Deviance_1()
            case _:
                pass
        
        temp = np.linspace(0, 2, 257)[:-1]

        prob_0 = np.exp(-np.square((temp - self.m_0)/(2*self.s_0)))/self.s_0
        prob_1 = np.exp(-np.square((temp - self.m_1)/(2*self.s_1)))/self.s_1
        prob_2 = np.exp(-np.square((temp - self.m_2)/(2*self.s_2)))/self.s_2
        
        self.prob_0 = prob_0/np.sqrt(2*np.pi)
        self.prob_1 = prob_1/np.sqrt(2*np.pi)
        self.prob_2 = prob_2/np.sqrt(2*np.pi)
        
        self.entry_0_d.config(state=tk.NORMAL)
        self.entry_0_d.delete(0, tk.END)
        self.entry_0_d.insert(0, f'{self.dev_0:.3f}')
        self.entry_0_d.config(state=tk.DISABLED)
        
        self.entry_1_d.config(state=tk.NORMAL)
        self.entry_1_d.delete(0, tk.END)
        self.entry_1_d.insert(0, f'{self.dev_1:.3f}')
        self.entry_1_d.config(state=tk.DISABLED)
        
        self.entry_2_d.config(state=tk.NORMAL)
        self.entry_2_d.delete(0, tk.END)
        self.entry_2_d.insert(0, f'{self.dev_2:.3f}')
        self.entry_2_d.config(state=tk.DISABLED)
        
        return None
    
    
    def _Deviance_0(self, *args, **kwargs) -> None:
        
        self.dev_0 = (self.p_0/256 - self.m_0)/self.s_0
        self.dev_1 = (self.p_1/2 - self.m_1)/self.s_1
        self.dev_2 = (self.p_2*2 - self.m_2)/self.s_2

        return None
    
    
    def _Deviance_1(self, *args, **kwargs) -> None:
        
        self.dev_0 = (self.p_0/256 - self.m_0)/self.s_0
        self.dev_1 = (self.p_1/(16*self.p_0) - self.m_1)/self.s_1
        self.dev_2 = (self.p_2*2 - self.m_2)/self.s_2
        
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
        
        r_0 = 256*(m_0 + 1.96*s_0)
        l_0 = 256*(m_0 - 1.96*s_0)
        
        r_1 = 100*(m_1 + 1.96*s_1)*2
        l_1 = 100*(m_1 - 1.96*s_1)*2
        
        r_2 = (m_2 + 1.96*s_2)/2
        l_2 = (m_2 - 1.96*s_2)/2
        
        temp = np.linspace(0, 2, 257)[:-1]
        
        plot_0 = self.plot_0
        canvas_0 = self.canvas_0
        
        plot_1 = self.plot_1
        canvas_1 = self.canvas_1
        
        plot_2 = self.plot_2
        canvas_2 = self.canvas_2
        
        plot_0.clear()
        plot_0.plot(256*temp, self.prob_0, color='blue')
        plot_0.axvline(self.p_0, color='red')
        plot_0.axvline(r_0, color='black', linestyle='dashed')
        plot_0.axvline(l_0, color='black', linestyle='dashed')
        plot_0.set_title("Radius Probability Function")
        plot_0.set_xlabel(r'Radius ($\AA$)')
        plot_0.set_ylabel(r'Probability Density')
        plot_0.grid()
        
        canvas_0.draw()
        
        plot_1.clear()
        plot_1.plot(100*(temp*2), self.prob_1, color='blue')
        plot_1.axvline(100*self.p_1, color='red')
        plot_1.axvline(r_1, color='black', linestyle='dashed')
        plot_1.axvline(l_1, color='black', linestyle='dashed')
        plot_1.set_title("Aspect Ratio Probability Function")
        plot_1.set_xlabel(r'Aspect Ratio (%)')
        plot_1.set_ylabel(r'Probability Density')
        plot_1.grid()
        
        canvas_1.draw()
        
        plot_2.clear()
        plot_2.plot(temp/2, self.prob_2, color='blue')
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
        
        r_0 = 256*(m_0 + 1.96*s_0)
        l_0 = 256*(m_0 - 1.96*s_0)
        
        r_1 = 16*(m_1 + 1.96*s_1)*self.p_0
        l_1 = 16*(m_1 - 1.96*s_1)*self.p_0
        
        r_2 = (m_2 + 1.96*s_2)/2
        l_2 = (m_2 - 1.96*s_2)/2
        
        temp = np.linspace(0, 2, 257)[:-1]
        
        plot_0 = self.plot_0
        canvas_0 = self.canvas_0
        
        plot_1 = self.plot_1
        canvas_1 = self.canvas_1
        
        plot_2 = self.plot_2
        canvas_2 = self.canvas_2
        
        plot_0.clear()
        plot_0.plot(256*temp, self.prob_0, color='blue')
        plot_0.axvline(self.p_0, color='red')
        plot_0.axvline(r_0, color='black', linestyle='dashed')
        plot_0.axvline(l_0, color='black', linestyle='dashed')
        plot_0.set_title("Radius Probability Function")
        plot_0.set_xlabel(r'Radius ($\AA$)')
        plot_0.set_ylabel(r'Probability Density')
        plot_0.grid()
        
        canvas_0.draw()
        
        plot_1.clear()
        plot_1.plot(16*temp*self.p_0, self.prob_1, color='blue')
        plot_1.axvline(self.p_1, color='red')
        plot_1.axvline(r_1, color='black', linestyle='dashed')
        plot_1.axvline(l_1, color='black', linestyle='dashed')
        plot_1.set_title("Aspect Ratio Probability Function")
        plot_1.set_xlabel(r'Aspect Ratio (%)')
        plot_1.set_ylabel(r'Probability Density')
        plot_1.grid()
        
        canvas_1.draw()
        
        plot_2.clear()
        plot_2.plot(temp/2, self.prob_2, color='blue')
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
        
        self._Classify()
        
        if not self.started:
            self._Fit()
            
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
        
        return None
    
    
    def _Simulate_as_0(self, *args, **kwargs) -> None:

        p_0 = self.entry_0.get()
        p_0 = float(p_0)
        self.p_0 = p_0
        
        p_1 = self.entry_1.get()
        p_1 = float(p_1)/100
        self.p_1 = p_1
        
        p_2 = self.entry_2.get()
        p_2 = float(p_2)
        self.p_2 = p_2
        
        p_3 = self.entry_3.get()
        p_3 = float(p_3)/100
        self.p_3 = p_3
        
        p_4 = self.entry_4.get()
        p_4 = float(p_4)/1000
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
        p_3 = float(p_3)/100
        self.p_3 = p_3
        
        p_4 = self.entry_4.get()
        p_4 = float(p_4)/1000
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
        
        n = 4096
        s = self.s
        
        scatterers = s.generate_scatterers(n=n)
            
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
            
            while f.readline()[0] not in "0123456789":
                continue
            
            while True:
                line = f.readline()
                
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
        X = 1 + np.log10(X)/2
        X = np.tanh(X)
        X = X[np.newaxis, :, np.newaxis]
        
        self.X = X
    
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
        
        qr = q_arr*r_g
        
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
        R_g = sqrt(-3*a)
        
        self.r_g_1 = R_g
        
        return None


def main(*args, **kwargs) -> int:
    
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
    
    return 0


if __name__ == '__main__':
    main()
