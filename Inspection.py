import os
import re
import shutil

import numpy as np
import tkinter as tk

from datetime import date

from tkinter import Button, StringVar, filedialog, Entry, Label

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MainApplication(tk.Frame):
    
    def __init__(self, parent, *args, **kwargs):
        
        tk.Frame.__init__(self, parent, *args, **kwargs)
        
        parent.title('SAXS GUI')
        parent.configure(bg='white')
        parent.geometry("540x640")
        parent.protocol("WM_DELETE_WINDOW", parent.quit())
        
        Button_load = Button(parent, text="Load Folder", command=self._LoadFolder)
        Button_load.place(height=30, width=110, x=14, y=14)
                
        label_Folder = Label(parent, text="Path:")
        label_Folder.place(height=30, width=60, x=140, y=14)
                
        folder_var = StringVar()
                
        Entry_folder = Entry(parent, textvariable=folder_var)
        Entry_folder.place(height=30, width=326, x=200, y=14)
        Entry_folder.config(state=tk.DISABLED)
                
        Button_clear = Button(parent, text="Clear", command=self._Clear)
        Button_clear.place(height=30, width=110, x=14, y=50)
        Button_clear.config(state=tk.DISABLED)
                
        label_File = Label(parent, text="File:")
        label_File.place(height=30, width=60, x=140, y=50)
                
        file_var = StringVar()
                
        Entry_file = Entry(parent, textvariable=file_var)
        Entry_file.place(height=30, width=326, x=200, y=50)
        Entry_file.config(state=tk.DISABLED)
                        
        Button_start = Button(parent, text="Start", command=self._Start)
        Button_start.place(height=30, width=80, x=14, y=86)
        Button_start.config(state=tk.DISABLED)
        
        Button_back = Button(parent, text="Back (B)", command=self._Back)
        Button_back.place(height=30, width=70, x=100, y=86)
        Button_back.config(state=tk.DISABLED)
        
        Button_skip = Button(parent, text="Skip (S)", command=self._Skip)
        Button_skip.place(height=30, width=80, x=176, y=86)
        Button_skip.config(state=tk.DISABLED)
                
        Button_disregard = Button(parent, text="Disregard (D)", command=self._Disregard)
        Button_disregard.place(height=30, width=90, x=262, y=86)
        Button_disregard.config(state=tk.DISABLED)
                
        Button_keep = Button(parent, text="Keep (K)", command=self._Keep)
        Button_keep.place(height=30, width=80, x=358, y=86)
        Button_keep.config(state=tk.DISABLED)
                
        label_Count = Label(parent, text="N/A")
        label_Count.place(height=30, width=82, x=444, y=86)

        Figure_scattering = Figure(figsize=(5, 4), dpi=85)
                
        Plot_scattering = Figure_scattering.add_subplot(1, 1, 1)
        Plot_scattering.set_title("Loaded Sample")
        Plot_scattering.set_xlabel(r'q ($\AA$)')
        Plot_scattering.set_ylabel('Scattering Intensity')
        Plot_scattering.set_xscale('log')
        Plot_scattering.set_yscale('log')
                
        canvas_scattering = FigureCanvasTkAgg(Figure_scattering, parent)
        canvas_scattering.get_tk_widget().place(height=400, width=512, x=14, y=128)
        
        Button_autosubtract = Button(parent, text="Autosubtraction", command=self._OpenAutosubtractionWindow)
        Button_autosubtract.place(height=30, width=502, x=14, y=548)
        Button_autosubtract.config(state=tk.DISABLED)
        
        Button_point_left = Button(parent, text="<", command=self._SelectPreviousPoint)
        Button_point_left.place(height=30, width=50, x=14, y=588)
        Button_point_left.config(state=tk.DISABLED)
        
        Button_point_right = Button(parent, text=">", command=self._SelectNextPoint)
        Button_point_right.place(height=30, width=50, x=70, y=588)
        Button_point_right.config(state=tk.DISABLED)
        
        point_I_var = StringVar()
        
        Entry_point_I = Entry(parent, textvariable=point_I_var)
        Entry_point_I.place(height=30, width=180, x=132, y=588)
        Entry_point_I.config(state=tk.DISABLED)
        
        Button_point_submit = Button(parent, text="Submit", command=self._SubmitSelectedPoint)
        Button_point_submit.place(height=30, width=90, x=324, y=588)
        Button_point_submit.config(state=tk.DISABLED)
        
        Button_point_delete = Button(parent, text="Delete", command=self._DeleteSelectedPoint)
        Button_point_delete.place(height=30, width=90, x=426, y=588)
        Button_point_delete.config(state=tk.DISABLED)
        
        parent.bind('s', self._Skip)
        parent.bind('d', self._Disregard)
        parent.bind('k', self._Keep)
        parent.bind('b', self._Back)
        
        self.file_loaded = False
        self.current_file_has_qI = False
        self.selected_point_index = None
        self.current_q_arr = np.array([])
        self.current_I_arr = np.array([])
        self.current_s_arr = np.array([])
        
        self.parent = parent
        self.Button_load = Button_load
        self.label_Folder = label_Folder
        self.folder_var = folder_var
        self.Entry_folder = Entry_folder
        self.Button_clear = Button_clear
        self.label_File = label_File
        self.file_var = file_var
        self.Entry_file = Entry_file
        self.Button_start = Button_start
        self.Button_back = Button_back
        self.Button_skip = Button_skip
        self.Button_disregard = Button_disregard
        self.Button_keep = Button_keep
        self.label_Count = label_Count
        self.Figure_scattering = Figure_scattering
        self.Plot_scattering = Plot_scattering
        self.canvas_scattering = canvas_scattering
        self.Button_autosubtract = Button_autosubtract
        self.Button_point_left = Button_point_left
        self.Button_point_right = Button_point_right
        self.point_I_var = point_I_var
        self.Entry_point_I = Entry_point_I
        self.Button_point_submit = Button_point_submit
        self.Button_point_delete = Button_point_delete

    
    def _Start(self) -> None:
        
        if self.folder_loaded:
            
            cwd = os.getcwd()
            
            base_path = os.path.join(cwd, 'Inspection')
            
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            today = str(date.today()).replace("-", "_")
            username = os.getlogin()

            path_name = f"{today}_{username}"
            attempt = 0
            
            while True:
                
                temp_name = f"{path_name}_{attempt}"
                temp_path = os.path.join(base_path, temp_name)
                
                if not os.path.exists(temp_path):
                    path = temp_path
                    
                    break
                else:
                    attempt += 1
            
            os.mkdir(path=path)
            
            skip_dir = os.path.join(path, 'Skip')
            disregard_dir = os.path.join(path, 'Disregard')
            keep_dir = os.path.join(path, 'Keep')
            
            os.mkdir(path=skip_dir)
            os.mkdir(path=disregard_dir)
            os.mkdir(path=keep_dir)
            
            comment = tk.simpledialog.askstring(
                title="Logging",
                prompt="Comment:"
            )
            
            comment_name = 'comment'
            comment_end = 'txt'
            comment_path = os.path.join(path, f'{comment_name}.{comment_end}')
            
            with open(comment_path, 'w') as f:
                f.write(comment)
            
            log_name = 'log'
            log_end = 'csv'
            log_path = os.path.join(path, f'{log_name}.{log_end}')
            
            self.path = path
            self.skip_dir = skip_dir
            self.disregard_dir = disregard_dir
            self.keep_dir = keep_dir
            self.log_path = log_path
            
            self.Button_back.config(state=tk.DISABLED)
            self.Button_skip.config(state=tk.NORMAL)
            self.Button_disregard.config(state=tk.NORMAL)
            self.Button_keep.config(state=tk.NORMAL)
            self.Button_start.config(state=tk.DISABLED)
            self.Button_clear.config(state=tk.NORMAL)
            
            self.label_Count.config(text=f'{self.count}/{self.max_count}')
            
            dir_list = self.dir_list
            count = self.count
            
            file = dir_list[count]
            self.current_file = file
            self._LoadFile()
        
        return None
    
    
    def _Record(self, label: str) -> None:
        
        source_path = self.source_path
        file = self.current_file
        log_path = self.log_path
        
        match label:
            case 's':
                temp_path = self.skip_dir
            case 'd':
                temp_path = self.disregard_dir
            case 'k':
                temp_path = self.keep_dir
            case _:
                pass
        
        origin = os.path.join(source_path, file)
        target = os.path.join(temp_path, file)
        if self.current_file_has_qI:
            self._save_qI_copy(
                origin=origin,
                target=target,
                q_arr=self.current_q_arr,
                I_arr=self.current_I_arr,
                s_arr=self.current_s_arr
            )
        elif label == 's':
            shutil.copyfile(origin, target)
        else:
            return None
        
        with open(log_path, "a") as f:
            f.write(f"{file},{label}\n")
        
        return None
    
    
    """
    The *args for the below three methods collects some unknown garbage
    input. I don't know what argument is being passed, but this seems to
    work, so don't remove this in the future.'
    """
    def _Skip(self, *args) -> None:
        
        if self.file_loaded:

            label = 's'
            
            self._Record(label=label)
            self.count += 1
            
            if self.count < self.max_count:
                self._LoadFile()
                self.label_Count.config(text=f'{self.count}/{self.max_count}')
            else:
                self.file_loaded = False
                self.current_file_has_qI = False
                self.Button_back.config(state=tk.NORMAL if self.count > 0 else tk.DISABLED)
                self.Button_skip.config(state=tk.DISABLED)
                self.Button_disregard.config(state=tk.DISABLED)
                self.Button_keep.config(state=tk.DISABLED)
                self._SetPointEditorStates()
                self.label_Count.config(text=f'{self.count}/{self.max_count}')
        
        return None
    
    
    def _Disregard(self, *args) -> None:
        
        if self.file_loaded and self.current_file_has_qI:
        
            label = 'd'
            
            self._Record(label=label)
            self.count += 1
            
            if self.count < self.max_count:
                self._LoadFile()
                self.label_Count.config(text=f'{self.count}/{self.max_count}')
            else:
                self.file_loaded = False
                self.current_file_has_qI = False
                self.Button_back.config(state=tk.NORMAL if self.count > 0 else tk.DISABLED)
                self.Button_skip.config(state=tk.DISABLED)
                self.Button_disregard.config(state=tk.DISABLED)
                self.Button_keep.config(state=tk.DISABLED)
                self._SetPointEditorStates()
                self.label_Count.config(text=f'{self.count}/{self.max_count}')
        
        return None
    
    
    def _Keep(self, *args) -> None:
        
        if self.file_loaded and self.current_file_has_qI:
        
            label = 'k'
            
            self._Record(label=label)
            self.count += 1
            
            if self.count < self.max_count:
                self._LoadFile()
                self.label_Count.config(text=f'{self.count}/{self.max_count}')
            else:
                self.file_loaded = False
                self.current_file_has_qI = False
                self.Button_back.config(state=tk.NORMAL if self.count > 0 else tk.DISABLED)
                self.Button_skip.config(state=tk.DISABLED)
                self.Button_disregard.config(state=tk.DISABLED)
                self.Button_keep.config(state=tk.DISABLED)
                self._SetPointEditorStates()
                self.label_Count.config(text=f'{self.count}/{self.max_count}')
        
        return None
    
    
    def _Back(self, *args) -> None:
        
        if self.count > 0 and self.dir_list:
            
            self.count -= 1
            file = self.dir_list[self.count]
            self._UndoRecord(file=file)
            self._LoadFile()
            self.label_Count.config(text=f'{self.count}/{self.max_count}')
        
        return None
    
    
    def _UndoRecord(self, file: str) -> None:
        
        for temp_path in (self.skip_dir, self.disregard_dir, self.keep_dir):
            target = os.path.join(temp_path, file)
            
            if os.path.exists(target):
                os.remove(target)
        
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
            
            with open(self.log_path, 'w') as f:
                for line in lines:
                    if not line.startswith(f'{file},'):
                        f.write(line)
        
        return None
    
    
    def _Clear(self) -> None:
        
        self.path = ''
        self.source_path = ''
        self.folder_loaded = False
        self.dir_list = []
        self.count = 0
        self.file_loaded = False
        self.current_file_has_qI = False
        self.selected_point_index = None
        self.current_q_arr = np.array([])
        self.current_I_arr = np.array([])
        self.current_s_arr = np.array([])
        
        self.Button_back.config(state=tk.DISABLED)
        self.Button_skip.config(state=tk.DISABLED)
        self.Button_disregard.config(state=tk.DISABLED)
        self.Button_keep.config(state=tk.DISABLED)
        self.Button_start.config(state=tk.DISABLED)
        self.Button_clear.config(state=tk.DISABLED)
        self.Button_autosubtract.config(state=tk.DISABLED)
        self._SetPointEditorStates()
        
        self.label_Count.config(text='N/A')
        
        Entry_folder = self.Entry_folder
        Entry_file = self.Entry_file
        
        Entry_folder.config(state=tk.NORMAL)
        Entry_folder.delete(0, tk.END)
        Entry_folder.config(state=tk.DISABLED)
        
        Entry_file.config(state=tk.NORMAL)
        Entry_file.delete(0, tk.END)
        Entry_file.config(state=tk.DISABLED)
                
        Plot_scattering = self.Plot_scattering
        canvas_scattering = self.canvas_scattering
        
        Plot_scattering.clear()
        Plot_scattering.set_title("Normalized Intensity")
        Plot_scattering.set_xlabel(r'q ($\AA$)')
        Plot_scattering.set_xscale('log')
        Plot_scattering.set_yscale('log')
        Plot_scattering.grid()
        
        canvas_scattering.draw()  
        
        return None
    
    
    def _LoadFile(self) -> None:
        
        dir_list = self.dir_list
        count = self.count
        source_path = self.source_path
        
        file = dir_list[count]
        file_path = os.path.join(source_path, file)
        
        self.current_file = file
        self.file_loaded = True
        
        Entry_file = self.Entry_file
        
        Entry_file.config(state=tk.NORMAL)
        Entry_file.delete(0, tk.END)
        Entry_file.insert(0, file)
        Entry_file.config(state=tk.DISABLED)
        
        Plot_scattering = self.Plot_scattering
        canvas_scattering = self.canvas_scattering
        
        Plot_scattering.clear()
        
        try:
            q_arr, I_arr, s_arr = self.get_qI(file=file_path)
        except ValueError:
            self.current_file_has_qI = False
            self.selected_point_index = None
            self.current_q_arr = np.array([])
            self.current_I_arr = np.array([])
            self.current_s_arr = np.array([])
            Plot_scattering.set_title("Invalid Column Format")
            Plot_scattering.text(
                0.5,
                0.5,
                "No 2- or 3-column numeric data found.\nOnly Skip is available.",
                ha='center',
                va='center',
                transform=Plot_scattering.transAxes
            )
        else:
            self.current_file_has_qI = True
            self.current_q_arr = q_arr.copy()
            self.current_I_arr = I_arr.copy()
            self.current_s_arr = s_arr.copy()
            self.selected_point_index = 0
            self._UpdatePointEntry()
            self._DrawCurrentPlot()
        
        self._SetFileButtonStates()
        Plot_scattering.grid()
        
        canvas_scattering.draw()    
        
        return None
    
    
    def _DrawCurrentPlot(self) -> None:
        
        Plot_scattering = self.Plot_scattering
        canvas_scattering = self.canvas_scattering
        
        Plot_scattering.clear()
        Plot_scattering.scatter(self.current_q_arr, self.current_I_arr, s=1)
        
        if self.selected_point_index is not None and len(self.current_q_arr) > 0:
            index = self.selected_point_index
            Plot_scattering.scatter(
                self.current_q_arr[index],
                self.current_I_arr[index],
                s=30,
                color='red'
            )
        
        Plot_scattering.set_title("Normalized Intensity")
        Plot_scattering.set_xlabel(r'q ($\AA$)')
        Plot_scattering.set_ylabel('Scattering Intensity')
        Plot_scattering.set_xscale('log')
        Plot_scattering.set_yscale('log')
        Plot_scattering.grid()
        
        canvas_scattering.draw()
        
        return None
    
    
    def _UpdatePointEntry(self) -> None:
        
        if self.selected_point_index is not None and len(self.current_I_arr) > 0:
            self.point_I_var.set(f'{self.current_I_arr[self.selected_point_index]:g}')
        else:
            self.point_I_var.set('')
        
        return None
    
    
    def _SelectPreviousPoint(self) -> None:
        
        if self.current_file_has_qI and self.selected_point_index is not None:
            self.selected_point_index = max(0, self.selected_point_index - 1)
            self._UpdatePointEntry()
            self._SetPointEditorStates()
            self._DrawCurrentPlot()
        
        return None
    
    
    def _SelectNextPoint(self) -> None:
        
        if self.current_file_has_qI and self.selected_point_index is not None:
            last_index = len(self.current_q_arr) - 1
            self.selected_point_index = min(last_index, self.selected_point_index + 1)
            self._UpdatePointEntry()
            self._SetPointEditorStates()
            self._DrawCurrentPlot()
        
        return None
    
    
    def _SubmitSelectedPoint(self) -> None:
        
        if self.current_file_has_qI and self.selected_point_index is not None:
            try:
                value = float(self.point_I_var.get())
            except ValueError:
                self._UpdatePointEntry()
                return None
            
            self.current_I_arr[self.selected_point_index] = value
            self._UpdatePointEntry()
            self._DrawCurrentPlot()
        
        return None
    
    
    def _DeleteSelectedPoint(self) -> None:
        
        if self.current_file_has_qI and self.selected_point_index is not None:
            index = self.selected_point_index
            self.current_q_arr = np.delete(self.current_q_arr, index)
            self.current_I_arr = np.delete(self.current_I_arr, index)
            self.current_s_arr = np.delete(self.current_s_arr, index)
            
            if len(self.current_q_arr) == 0:
                self.current_file_has_qI = False
                self.selected_point_index = None
                self._UpdatePointEntry()
                self._SetFileButtonStates()
                
                self.Plot_scattering.clear()
                self.Plot_scattering.set_title("No Data Points")
                self.Plot_scattering.text(
                    0.5,
                    0.5,
                    "All data points were deleted.\nOnly Skip is available.",
                    ha='center',
                    va='center',
                    transform=self.Plot_scattering.transAxes
                )
                self.Plot_scattering.grid()
                self.canvas_scattering.draw()
                
                return None
            
            self.selected_point_index = min(index, len(self.current_q_arr) - 1)
            self._UpdatePointEntry()
            self._SetFileButtonStates()
            self._DrawCurrentPlot()
        
        return None
    
    
    def _OpenAutosubtractionWindow(self) -> None:
        
        if not self.current_file_has_qI:
            return None
        
        window = tk.Toplevel(self.parent)
        window.title('Autosubtraction')
        window.geometry("900x420")
        
        self.autosub_window = window
        self.autosub_loaded_buffer = False
        self.autosub_subtracted = False
        self.autosub_q_crit = 0.2
        self.autosub_buffer_file = ''
        self.autosub_buffer_q_arr = np.array([])
        self.autosub_buffer_I_arr = np.array([])
        self.autosub_buffer_s_arr = np.array([])
        self.autosub_q_arr = np.array([])
        self.autosub_I_arr = np.array([])
        self.autosub_s_arr = np.array([])
        
        reg = window.register(self._ValidateFloatEntry)
        
        Button_buffer = Button(window, text="Buffer", command=self._LoadAutosubtractionBuffer)
        Button_buffer.place(height=44, width=170, x=20, y=20)
        
        Button_subtract = Button(window, text="Subtract", command=self._RunAutosubtraction)
        Button_subtract.place(height=44, width=170, x=20, y=76)
        Button_subtract.config(state=tk.DISABLED)
        
        Button_cutoff = Button(window, text="Cutoff", command=self._UpdateAutosubtractionCutoff)
        Button_cutoff.place(height=44, width=80, x=20, y=132)
        
        autosub_q_var = StringVar()
        autosub_q_var.set(f'{self.autosub_q_crit:g}')
        
        Entry_cutoff = Entry(window, textvariable=autosub_q_var)
        Entry_cutoff.place(height=44, width=82, x=108, y=132)
        Entry_cutoff.config(validate="key", validatecommand=(reg, '%P'))
        
        Button_export = Button(window, text="Export", command=self._ExportAutosubtraction)
        Button_export.place(height=44, width=170, x=20, y=188)
        Button_export.config(state=tk.DISABLED)
        
        label_status = Label(window, text='Load buffer')
        label_status.place(height=44, width=170, x=20, y=244)
        
        Figure_autosub_raw = Figure(figsize=(4, 4), dpi=80)
        Plot_autosub_raw = Figure_autosub_raw.add_subplot(1, 1, 1)
        
        canvas_autosub_raw = FigureCanvasTkAgg(Figure_autosub_raw, window)
        canvas_autosub_raw.get_tk_widget().place(height=360, width=330, x=220, y=30)
        
        Figure_autosub_sub = Figure(figsize=(4, 4), dpi=80)
        Plot_autosub_sub = Figure_autosub_sub.add_subplot(1, 1, 1)
        
        canvas_autosub_sub = FigureCanvasTkAgg(Figure_autosub_sub, window)
        canvas_autosub_sub.get_tk_widget().place(height=360, width=330, x=560, y=30)
        
        self.Button_autosub_buffer = Button_buffer
        self.Button_autosub_subtract = Button_subtract
        self.Button_autosub_export = Button_export
        self.autosub_q_var = autosub_q_var
        self.label_autosub_status = label_status
        self.Figure_autosub_raw = Figure_autosub_raw
        self.Figure_autosub_sub = Figure_autosub_sub
        self.Plot_autosub_raw = Plot_autosub_raw
        self.Plot_autosub_sub = Plot_autosub_sub
        self.canvas_autosub_raw = canvas_autosub_raw
        self.canvas_autosub_sub = canvas_autosub_sub
        
        self._DrawAutosubtractionRawPlot()
        self._DrawAutosubtractionSubtractedPlot()
        
        return None
    
    
    def _ValidateFloatEntry(self, input_: str, *args, **kwargs) -> bool:
        
        if input_ == '':
            return True
        
        try:
            float(input_)
        except ValueError:
            return False
        
        return True
    
    
    def _LoadAutosubtractionBuffer(self) -> None:
        
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select a Buffer File"
        )
        
        if filename:
            try:
                q_arr, I_arr, s_arr = self.get_qI(file=filename)
            except ValueError:
                self.autosub_loaded_buffer = False
                self.Button_autosub_subtract.config(state=tk.DISABLED)
                self.Button_autosub_export.config(state=tk.DISABLED)
                self.label_autosub_status.config(text='Invalid buffer')
                return None
            
            self.autosub_buffer_file = filename
            self.autosub_buffer_q_arr = q_arr
            self.autosub_buffer_I_arr = I_arr
            self.autosub_buffer_s_arr = s_arr
            self.autosub_loaded_buffer = True
            self.autosub_subtracted = False
            
            self.Button_autosub_buffer.config(text=os.path.basename(filename))
            self.Button_autosub_subtract.config(state=tk.NORMAL)
            self.Button_autosub_export.config(state=tk.DISABLED)
            self.label_autosub_status.config(text='Buffer loaded')
            self._DrawAutosubtractionRawPlot()
            self._DrawAutosubtractionSubtractedPlot()
        
        return None
    
    
    def _UpdateAutosubtractionCutoff(self) -> None:
        
        try:
            q_crit = float(self.autosub_q_var.get())
        except ValueError:
            self.autosub_q_var.set(f'{self.autosub_q_crit:g}')
            return None
        
        self.autosub_q_crit = q_crit
        self.autosub_subtracted = False
        self.Button_autosub_export.config(state=tk.DISABLED)
        self.label_autosub_status.config(text=f'Cutoff {q_crit:g}')
        self._DrawAutosubtractionRawPlot()
        self._DrawAutosubtractionSubtractedPlot()
        
        return None
    
    
    def _GetAutosubtractionBufferOnRawGrid(self) -> tuple[np.ndarray, np.ndarray]:
        
        raw_q = self.current_q_arr
        raw_min = np.min(raw_q)
        raw_max = np.max(raw_q)
        
        buffer_q = self.autosub_buffer_q_arr
        buffer_I = self.autosub_buffer_I_arr
        buffer_s = self.autosub_buffer_s_arr
        
        order = np.argsort(buffer_q)
        buffer_q = buffer_q[order]
        buffer_I = buffer_I[order]
        buffer_s = buffer_s[order]
        
        if raw_min < buffer_q[0] or raw_max > buffer_q[-1]:
            raise ValueError("Buffer q range does not cover the raw data q range")
        
        interp_I = np.interp(raw_q, buffer_q, buffer_I)
        interp_s = np.interp(raw_q, buffer_q, buffer_s)
        
        return interp_I, interp_s
    
    
    def _RunAutosubtraction(self) -> None:
        
        if not self.current_file_has_qI or not self.autosub_loaded_buffer:
            return None
        
        self._UpdateAutosubtractionCutoff()
        
        try:
            buffer_I, buffer_s = self._GetAutosubtractionBufferOnRawGrid()
        except ValueError as error:
            self.label_autosub_status.config(text=str(error))
            return None
        
        mask = self.current_q_arr > self.autosub_q_crit
        
        if np.count_nonzero(mask) == 0:
            self.label_autosub_status.config(text='No q above cutoff')
            return None
        
        sum_01 = np.sum(self.current_I_arr[mask]*buffer_I[mask])
        sum_11 = np.sum(np.square(buffer_I[mask]))
        
        if sum_11 == 0:
            self.label_autosub_status.config(text='Zero buffer scale')
            return None
        
        alpha = sum_01/sum_11
        sub_I = self.current_I_arr - alpha*buffer_I
        positive = sub_I[sub_I > 0]
        
        if len(positive) > 0:
            sub_I[sub_I <= 0] = np.min(positive)
        else:
            raw_positive = self.current_I_arr[self.current_I_arr > 0]
            floor = np.min(raw_positive)*1e-6 if len(raw_positive) > 0 else np.finfo(float).tiny
            sub_I[:] = floor
        
        self.autosub_alpha = alpha
        self.autosub_q_arr = self.current_q_arr.copy()
        self.autosub_I_arr = sub_I
        self.autosub_s_arr = np.sqrt(np.square(self.current_s_arr) + np.square(alpha*buffer_s))
        self.autosub_subtracted = True
        
        self.Button_autosub_export.config(state=tk.NORMAL)
        self.label_autosub_status.config(text=f'Alpha {alpha:g}')
        self._DrawAutosubtractionSubtractedPlot()
        
        return None
    
    
    def _ExportAutosubtraction(self) -> None:
        
        if not self.autosub_subtracted:
            return None
        
        self.current_q_arr = self.autosub_q_arr.copy()
        self.current_I_arr = self.autosub_I_arr.copy()
        self.current_s_arr = self.autosub_s_arr.copy()
        self.selected_point_index = 0 if len(self.current_q_arr) > 0 else None
        
        self._UpdatePointEntry()
        self._SetFileButtonStates()
        self._DrawCurrentPlot()
        self.label_autosub_status.config(text='Exported')
        
        return None
    
    
    def _DrawAutosubtractionRawPlot(self) -> None:
        
        plot = self.Plot_autosub_raw
        plot.clear()
        plot.plot(self.current_q_arr, self.current_I_arr, label='Raw')
        
        if self.autosub_loaded_buffer:
            plot.plot(self.autosub_buffer_q_arr, self.autosub_buffer_I_arr, label='Buffer')
            plot.legend()
        
        plot.axvline(x=self.autosub_q_crit, color='r')
        plot.set_title("Raw")
        plot.set_xlabel(r'q ($\AA^{-1}$)')
        plot.set_ylabel('Scattering Intensity')
        plot.set_xscale('log')
        plot.set_yscale('log')
        plot.grid()
        
        self.canvas_autosub_raw.draw()
        
        return None
    
    
    def _DrawAutosubtractionSubtractedPlot(self) -> None:
        
        plot = self.Plot_autosub_sub
        plot.clear()
        
        if self.autosub_subtracted:
            plot.plot(self.autosub_q_arr, self.autosub_I_arr)
        
        plot.set_title("Subtracted")
        plot.set_xlabel(r'q ($\AA^{-1}$)')
        plot.set_ylabel('Scattering Intensity')
        plot.set_xscale('log')
        plot.set_yscale('log')
        plot.grid()
        
        self.canvas_autosub_sub.draw()
        
        return None
    
    
    def _LoadFolder(self) -> None:
        
        root = self.parent
        
        root.path = filedialog.askdirectory(
            initialdir=os.getcwd(), 
            title="Load Folder"
        )
        
        source_path = root.path
        
        if source_path:
            
            Entry_folder = self.Entry_folder
            
            Entry_folder.config(state=tk.NORMAL)
            Entry_folder.delete(0, tk.END)
            Entry_folder.insert(0, source_path)
            Entry_folder.config(state=tk.DISABLED)
            
            self.Button_start.config(state=tk.NORMAL)
            self.Button_clear.config(state=tk.NORMAL)
            
            self.Entry_folder = Entry_folder
            self.source_path = source_path
            self.folder_loaded = True
            self.dir_list = os.listdir(source_path)
            self.count = 0
            self.max_count = len(self.dir_list)
                
        return None
    
    
    def _SetFileButtonStates(self) -> None:
        
        if self.file_loaded:
            self.Button_back.config(state=tk.NORMAL if self.count > 0 else tk.DISABLED)
            self.Button_skip.config(state=tk.NORMAL)
            self.Button_disregard.config(state=tk.NORMAL if self.current_file_has_qI else tk.DISABLED)
            self.Button_keep.config(state=tk.NORMAL if self.current_file_has_qI else tk.DISABLED)
            self.Button_autosubtract.config(state=tk.NORMAL if self.current_file_has_qI else tk.DISABLED)
        else:
            self.Button_back.config(state=tk.DISABLED)
            self.Button_skip.config(state=tk.DISABLED)
            self.Button_disregard.config(state=tk.DISABLED)
            self.Button_keep.config(state=tk.DISABLED)
            self.Button_autosubtract.config(state=tk.DISABLED)
        
        self._SetPointEditorStates()
        
        return None
    
    
    def _SetPointEditorStates(self) -> None:
        
        has_selected_point = (
            self.file_loaded
            and self.current_file_has_qI
            and self.selected_point_index is not None
            and len(self.current_q_arr) > 0
        )
        
        if has_selected_point:
            self.Button_point_left.config(
                state=tk.NORMAL if self.selected_point_index > 0 else tk.DISABLED
            )
            self.Button_point_right.config(
                state=tk.NORMAL if self.selected_point_index < len(self.current_q_arr) - 1 else tk.DISABLED
            )
            self.Entry_point_I.config(state=tk.NORMAL)
            self.Button_point_submit.config(state=tk.NORMAL)
            self.Button_point_delete.config(state=tk.NORMAL)
        else:
            self.Button_point_left.config(state=tk.DISABLED)
            self.Button_point_right.config(state=tk.DISABLED)
            self.Entry_point_I.config(state=tk.DISABLED)
            self.Button_point_submit.config(state=tk.DISABLED)
            self.Button_point_delete.config(state=tk.DISABLED)
        
        if not has_selected_point:
            self.point_I_var.set('')
        
        return None
    

    def _validate_line(self, text: str, repeat: int=3):
        float_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        sep_pattern = r"(?:\s*,\s*|\s+)"
        pattern = rf"^\s*(?:{float_pattern}{sep_pattern}){{{repeat - 1}}}{float_pattern}\s*$"
        
        return re.fullmatch(pattern, text) is not None
    
    
    def _get_data_separator(self, file: str) -> str:
        
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if self._validate_line(text=line, repeat=3) or self._validate_line(text=line, repeat=2):
                    if ',' in line:
                        return ','
                    if '\t' in line:
                        return '\t'
                    return ' '
        
        return ' '
    
    
    def _save_qI_copy(
        self,
        origin: str,
        target: str,
        q_arr: np.ndarray | None=None,
        I_arr: np.ndarray | None=None,
        s_arr: np.ndarray | None=None
    ) -> None:
        
        if q_arr is None or I_arr is None or s_arr is None:
            q_arr, I_arr, s_arr = self.get_qI(file=origin)
        
        sep = self._get_data_separator(file=origin)
        
        with open(target, 'w') as f:
            f.write(sep.join(('q', 'I(q)', 's(q)')) + '\n')
            
            for q, I, s in zip(q_arr, I_arr, s_arr):
                f.write(sep.join((f'{q:g}', f'{I:g}', f'{s:g}')) + '\n')
        
        return None
    
    
    def get_qI(self, file: str, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        filenameshort = os.path.basename(file)
        end = filenameshort[-3:]
        
        if 'button' in kwargs:
            _button = kwargs['button']
            _button.configure(text=filenameshort)
                    
        temp = list()
        
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                if self._validate_line(text=line, repeat=3):
                    q, I, s = re.split(r"\s*,\s*|\s+", line)
                    temp.append((float(q), float(I), float(s)))
                elif self._validate_line(text=line, repeat=2):
                    q, I = re.split(r"\s*,\s*|\s+", line)
                    temp.append((float(q), float(I), np.square(float(I))))
        
        temp = np.array(temp, dtype=float)
        
        if temp.size == 0:
            raise ValueError(f"No 2- or 3-column numeric data found in {file}")
        
        qs = temp[:, 0]
        Is = temp[:, 1]

        ss = temp[:, 2]
        
        return qs, Is, ss


def main(*args, **kwargs) -> int:
    
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
    
    return 0


if __name__ == '__main__':
    main()
