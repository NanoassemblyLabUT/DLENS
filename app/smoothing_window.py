"""Smoothing popup window mixin."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import Button, Entry, StringVar

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SmoothingWindowMixin:
    """Methods that own the smoothing popup workflow."""

    def _Smoothen(self) -> None:
        self._Pop_Up_1()
        return None
    

    def _Pop_Up_1(self, *args, **kwargs) -> None:

        self.pop = tk.Toplevel()
        self._Set_Smo_UI()

        return None

    def _Set_Smo_UI(self, *args, **kwargs) -> None:

        parent = self.pop

        parent.title('Smoothing')
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

