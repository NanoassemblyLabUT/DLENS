"""Auto-subtraction popup window mixin."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import Button, Entry, Label, StringVar, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from app.widgets import ToolTip
from core.background import auto_subtract_background


class BackgroundSubtractionWindowMixin:
    """Methods that own the auto-subtraction popup workflow."""

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
        button_raw.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        ToolTip(button_raw, "Load the raw data file for subtraction.")

        button_back = Button(parent, text="Load Buffer Data", command=self._Sub_Load_1)
        button_back.place(height=height, width=width, x=dx, y=4 * dy)
        button_back.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
            self.button_simulate.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

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
            self.sub_button_sub.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.sub_button_clear.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.sub_button_use.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        elif self.sub_loaded_0 or self.sub_loaded_1:
            self.sub_button_sub.config(state=tk.DISABLED, bg="Light grey")
            self.sub_button_clear.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
            result = auto_subtract_background(
                raw_q=self.qs_0,
                raw_i=self.Is_0,
                raw_s=self.ss_0,
                background_q=self.qs_1,
                background_i=self.Is_1,
                background_s=self.ss_1,
                q_crit=q_crit,
            )

            self.alpha = result.scale_factor
            self.qs_2 = result.q
            self.Is_2 = result.intensity
            self.ss_2 = result.sigma
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
