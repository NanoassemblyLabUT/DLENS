import os
import sys

# --- WINDOWS GUI SILENT CRASH FIX ---
# Prevents tqdm and print() from crashing the app when the console is hidden
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
# ------------------------------------

import shutil
import multiprocessing

import numpy as np
import tkinter as tk

from tkinter import filedialog, Entry, Label

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from core.config import reference_q_grid
from core.data_io import create_run_directory, load_saxs_profile
from core.metrics import guinier_radius, mean_log_squared_error
from core.preprocessing import interpolate_for_classifier, prepare_model_input
from inference.model_loader import load_all_models
from inference.pipeline import (
    classify_profile_hierarchical, 
    prepare_classifier_input as prepare_shape_classifier_input
)
from app.widgets import ToolTip
from app.background_window import BackgroundSubtractionWindowMixin
from app.layout import LayoutMixin
from app.probability_window import ProbabilityWindowMixin
from app.smoothing_window import SmoothingWindowMixin
from app.subclass_probability_window import SubclassProbabilityWindowMixin
from inference.shape_registry import PredictedParameters, SHAPES, SHAPES_BY_CLASS, SHAPES_BY_DISPLAY_NAME


FAMILY_KEY_TO_DISPLAY = {"spheroid": "Isotropic", "cylinder": "Anisotropic"}
FAMILY_DISPLAY_TO_KEY = {display: key for key, display in FAMILY_KEY_TO_DISPLAY.items()}
FAMILY_KEY_TO_CLASS_ID = {"spheroid": 0, "cylinder": 1}

"""
The Debye module is a file that works alongside this file.
"""


class MainApplication(LayoutMixin, ProbabilityWindowMixin, SubclassProbabilityWindowMixin, BackgroundSubtractionWindowMixin, SmoothingWindowMixin, tk.Frame):

    def __init__(self, parent, *args, **kwargs) -> None:

        self.parent = parent

        self._Setting()
        self._Layout()
        self._AddDynamicWidgets()
        self._LoadModels()

        return None

    def _Setting(self, *args, **kwargs) -> None:

        self.q_log_arr, self.q_arr = reference_q_grid()

        if getattr(sys, 'frozen', False):
            bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            if bundle_dir.endswith('MacOS') or bundle_dir.endswith('Frameworks'):
                resources_dir = os.path.join(os.path.dirname(bundle_dir), 'Resources')
                if os.path.exists(resources_dir):
                    bundle_dir = resources_dir
            self.cwd = bundle_dir
            os.chdir(self.cwd)  
        else:
            self.cwd = os.getcwd()

        self.last_opened_dir = self.cwd 
        self.auto_export_config = None 
        self.current_batch_mode = "click"

        self.log_file = None
        self.base_path = None
        self.working_dir = None
        self.log_path = None
        
        self.origin = ""
        self.file_path = ""

        self.file_loaded = False
        self.folder_loaded = False
        self.fitted = False
        self.started = False

        self.shape = None
        self._class = None
        self._family_class = None
        self.subclass_id = None
        self.count = -1

        return None

    def _AddDynamicWidgets(self) -> None:
        self.brute_force_var = tk.BooleanVar(value=False)
        self.check_brute_force = tk.Checkbutton(self.parent, text="Auto-fit all subclasses (Slower)", variable=self.brute_force_var)
        self.check_brute_force.place(x=660, y=16.5 * self.dy)
        try: ToolTip(self.check_brute_force, "Bypasses the AI subclass prediction and simulates all subclasses to find the lowest mMSLE")
        except: pass

        self.button_update_export = tk.Button(self.parent, text="Update Export", command=self._ForceUpdateExport)
        self.button_update_export.place(height=30, width=140, x=680, y=18.5 * self.dy)
        self.button_update_export.config(state=tk.DISABLED, bg="Light grey")
        try: ToolTip(self.button_update_export, "Apply manual changes and update the exported batch file")
        except: pass
        
        return None

    def _LoadModels(self, *args, **kwargs) -> None:

        models = load_all_models(self.cwd)

        self.model_qr = models["qr"]
        self.model_family_cl = models["family_classifier"]
        self.model_cl = models["classifier"]
        self.model_subclassifiers = models["subclassifiers"]
        self.models = models

        return None

    def _Callback(self, input_: str, *args, **kwargs) -> bool:

        try:
            if input_ == '':
                return True
            float(input_)
        except ValueError:
            return False
        return True

    def _EnsureExportDirectory(self) -> None:
        
        if self.working_dir is not None and self.log_path is not None:
            return None

        run_paths = create_run_directory(self.cwd)
        self.log_file = run_paths["log_file"]
        self.base_path = run_paths["base_path"]
        self.working_dir = run_paths["working_dir"]
        self.log_path = run_paths["log_path"]
        return None

    def _SubclassDisplayNames(self, family_key: str | None = None) -> list[str]:
        return [
            spec.display_name
            for spec in SHAPES.values()
            if spec.model_key is not None and (family_key is None or spec.family_key == family_key)
        ]

    def _RefreshSubclassMenu(self, family_key: str | None = None, selected: str | None = None) -> None:
        names = self._SubclassDisplayNames(family_key)
        menu = self.drop_methods["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name, command=tk._setit(self.select_shapes, name, self._Drop_Fit))

        if selected in names:
            self.select_shapes.set(selected)
        elif names:
            self.select_shapes.set(names[0])
        else:
            self.select_shapes.set("Subclass")

        return None

    def _ApplyShapeSpec(self, spec, *, sync_selectors: bool = True) -> None:
        self.shape = spec.display_name
        self._class = spec.class_id
        self._family_class = FAMILY_KEY_TO_CLASS_ID[spec.family_key]
        self.family_key = spec.family_key
        self.family_class_id = self._family_class
        self.subclass_id = spec.subclass_id
        self.shape_spec = spec
        self.old_mode = spec.display_name
        self.new_mode = spec.display_name

        if sync_selectors and hasattr(self, "select_family"):
            self.select_family.set(FAMILY_KEY_TO_DISPLAY[spec.family_key])
            self._RefreshSubclassMenu(spec.family_key, selected=spec.display_name)

        return None

    def _Drop_Fit(self, input_: str, *args, **kwargs) -> None:

        spec = SHAPES_BY_DISPLAY_NAME.get(input_)
        if spec is None:
            return None

        self._ApplyShapeSpec(spec)
        self._Reconfigure()

        self.button_clear.configure(state=tk.NORMAL)
        self.drop_methods.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.drop_families.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

        if hasattr(self, "I_arr"):
            self._Fit()
            self._DisplaySubclassProbabilities()
        elif self.started:
            self.button_export.configure(state=tk.NORMAL)
            self.button_simulate.configure(state=tk.NORMAL)
            self.button_visualize.configure(state=tk.NORMAL)
        else:
            self.button_export.configure(state=tk.DISABLED)
            self.button_simulate.configure(state=tk.NORMAL)
            self.button_visualize.configure(state=tk.DISABLED)

        return None

    def _Drop_Family(self, input_: str, *args, **kwargs) -> None:

        family_key = FAMILY_DISPLAY_TO_KEY.get(input_)
        if family_key is None:
            return None

        names = self._SubclassDisplayNames(family_key)
        current_spec = getattr(self, "shape_spec", None)
        if current_spec is not None and current_spec.family_key == family_key:
            selected = current_spec.display_name
        else:
            selected = names[0] if names else None

        self._RefreshSubclassMenu(family_key, selected=selected)
        if selected is not None:
            self._Drop_Fit(selected)

        return None

    def _SubclassProbabilityRows(self) -> list[tuple[str, float | None]]:
        spec = getattr(self, "shape_spec", None)
        if spec is None:
            return []

        shapes = [
            shape
            for shape in SHAPES.values()
            if shape.model_key is not None and shape.family_key == spec.family_key
        ]
        shapes.sort(key=lambda shape: -1 if shape.subclass_id is None else shape.subclass_id)

        probabilities = {}
        subclassifier = self.model_subclassifiers.get(spec.family_key)
        if hasattr(self, "I_arr") and subclassifier is not None and hasattr(subclassifier, "predict_proba"):
            classifier_input = prepare_shape_classifier_input(self.I_arr)
            row = np.asarray(subclassifier.predict_proba(classifier_input))[0]
            class_ids = getattr(subclassifier, "classes_", range(len(row)))
            probabilities = {int(class_id): float(probability) for class_id, probability in zip(class_ids, row)}

        return [(shape.display_name, probabilities.get(shape.subclass_id)) for shape in shapes]

    def _Change_Mode(self, *args, **kwargs) -> None:
        self._Drop_Fit(self.select_shapes.get())
        return None

    def _ClearEntries(self, *args, **kwargs) -> None:

        if self._ProbabilityEntries():
            for entries in self._ProbabilityEntries():
                for entry in entries:
                    self._SetEntryText(entry, "", disabled=True)
        self.entry_0.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_0.delete(0, tk.END)
        self.entry_0.config(state=tk.DISABLED, bg="Light grey")

        self.entry_1.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_1.delete(0, tk.END)
        self.entry_1.config(state=tk.DISABLED, bg="Light grey")

        self.entry_2.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_2.delete(0, tk.END)
        self.entry_2.config(state=tk.DISABLED, bg="Light grey")

        self.entry_3.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_3.delete(0, tk.END)
        self.entry_3.config(state=tk.DISABLED, bg="Light grey")

        self.entry_4.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_4.delete(0, tk.END)
        self.entry_4.config(state=tk.DISABLED, bg="Light grey")

        self.entry_5.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_5.delete(0, tk.END)
        self.entry_5.config(state=tk.DISABLED, bg="Light grey")

        self.entry_6.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_6.delete(0, tk.END)
        self.entry_6.config(state=tk.DISABLED, bg="Light grey")

        self.entry_7.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_7.delete(0, tk.END)
        self.entry_7.config(state=tk.DISABLED, bg="Light grey")

        self.entry_8.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_8.delete(0, tk.END)
        self.entry_8.config(state=tk.DISABLED, bg="Light grey")

        self.entry_9.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_9.delete(0, tk.END)
        self.entry_9.config(state=tk.DISABLED, bg="Light grey")

        self.Entry_MSLE.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.Entry_MSLE.delete(0, tk.END)
        self.Entry_MSLE.config(state=tk.DISABLED, disabledbackground="Light grey", disabledforeground="black")

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
        self.plot_s.figure.tight_layout() 
        self.canvas_s.draw()

        if self._ProbabilityEntries():
            for plot, canvas in ((self.plot_0, self.canvas_0), (self.plot_1, self.canvas_1), (self.plot_2, self.canvas_2)):
                plot.clear()
                plot.set_ylabel('Probability Density')
                plot.grid()
                plot.figure.tight_layout() 
                canvas.draw()
        return None

    def _Clear(self, *args, **kwargs) -> None:
        self.file_loaded = False
        self.started = False
        self.fitted = False

        self._ToggleFeatures()
        self._ClearEntries()
        self._ClearButtons()

        self.select_family.set("Main Class")
        self.select_shapes.set("Subclass")
        self._RefreshSubclassMenu()
        self.drop_families.config(state=tk.DISABLED, bg="Light grey")
        self.drop_methods.config(state=tk.DISABLED, bg="Light grey")
        self.button_subclass_probability.config(state=tk.DISABLED, bg="Light grey")
        self._CloseSubclassProbabilityWindow()

        if not self.folder_loaded:
            self.label_count.config(text='N/A')
        return None

    def _Change_File(self, forward: bool, *args, **kwargs) -> None:

        if forward:
            self.count = min(self.count + 1, self.max_count - 1)
        else:
            self.count = max(self.count - 1, 0)

        if self.count == self.max_count - 1:
            self.button_forward.config(state=tk.DISABLED, bg="Light grey")
            self.button_backward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        elif self.count == 0:
            self.button_forward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_backward.config(state=tk.DISABLED, bg="Light grey")
        else:
            self.button_forward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_backward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

        filenameshort = self.dir_list[self.count]
        filename = os.path.join(self.source_path, filenameshort)

        self.origin = self.source_path
        self.file_path = filename

        self._PrepareFile()
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

        root = self.parent

        root.filename = filedialog.askopenfilename(
            initialdir=self.last_opened_dir,
            title="Select A File",
            filetypes=[("SAXS Data", "*.csv *.txt *.dat *.out *.iq *.int *.fit"), ("All Files", "*.*")]
        )

        filename = root.filename

        if filename:
            self.auto_export_config = None 

            filenameshort = os.path.basename(filename)
            name_len = len(filenameshort)
            folder_name = filename[:-name_len]

            self.last_opened_dir = folder_name

            self.file_loaded = True
            self.folder_loaded = False
            self.origin = folder_name
            self.file_path = filename

            self._PrepareFile()

        self.update_status("File loaded successfully.")

        return None

    def _PrepareFile(self, *args, **kwargs) -> None:
        """Safely loads, prepares, and fits the file. Prevents silent crashing on ML failure."""
        self._Clear()
        self.button_simulate.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        
        try:
            self.get_qI()
            self._Draw_qI()
            self._Classify()
            self._Fit()
            self.update_status("File loaded successfully.")
        except Exception as e:
            error_msg = str(e)
            print(f"Error during file preparation: {error_msg}")
            self.update_status(f"Prediction Error: {error_msg}")

        return None

    def _LoadFolder(self, *args, **kwargs) -> None:
        self.update_status("Loading folder...")

        root = self.parent

        root.path = filedialog.askdirectory(
            initialdir=self.last_opened_dir,
            title="Load Folder"
        )

        source_path = root.path

        if source_path:
            self.last_opened_dir = source_path

            self._Clear()

            self.source_path = source_path
            self.file_loaded = False
            self.folder_loaded = True
            
            valid_exts = ('.csv', '.txt', '.dat', '.out', '.iq', '.int', '.fit')
            self.dir_list = [f for f in os.listdir(source_path) if not f.startswith('.') and f.lower().endswith(valid_exts)]
            self.count = -1
            self.max_count = len(self.dir_list)

            if self.max_count == 0:
                self.update_status("Error: No valid data files found in folder.")
                self.folder_loaded = False
                return

            self.button_file.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_clear.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_forward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_backward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

            self.parent.after(250, self._BatchExportSetup)

        else:
            self.folder_loaded = False
            self.update_status("Folder load cancelled.")

        return None

    def _BatchExportSetup(self) -> None:
        """Pop up window to configure automatic saving when clicking through a folder"""
        self.update_status("Opening folder auto-export options...")

        pop = tk.Toplevel(self.parent)
        pop.title("Folder Auto-Export Setup")
        pop.geometry("420x470")
        pop.resizable(False, False)
        pop.geometry(f"+{self.parent.winfo_rootx() + 50}+{self.parent.winfo_rooty() + 50}")

        tk.Label(pop, text="Where should results for this folder be saved?").place(x=16, y=16)
        
        default_out = os.path.join(self.source_path, "DLENS_Output")
        self.batch_dir_var = tk.StringVar(value=default_out)
        entry_dir = tk.Entry(pop, textvariable=self.batch_dir_var, state='readonly')
        entry_dir.place(x=16, y=40, width=300)
        
        def _browse_dir():
            chosen = filedialog.askdirectory(initialdir=self.source_path)
            if chosen:  
                self.batch_dir_var.set(chosen)
                
        tk.Button(pop, text="Browse", command=_browse_dir).place(height=26, width=70, x=325, y=38)

        tk.Label(pop, text="Select files to auto-generate for EACH sample:").place(x=16, y=80)

        self.b_opt_log = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Append to Parameter Log (Record.csv)", variable=self.b_opt_log).place(x=16, y=105)

        self.b_opt_raw = tk.BooleanVar(value=False)
        tk.Checkbutton(pop, text="Copy Original SAXS Data", variable=self.b_opt_raw).place(x=16, y=130)

        self.b_opt_sim_csv = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Simulated SAXS Profile (.csv)", variable=self.b_opt_sim_csv).place(x=16, y=155)

        self.b_opt_sim_fig = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="SAXS Profile Plot (.png)", variable=self.b_opt_sim_fig).place(x=16, y=180)

        self.b_opt_prob_csv = tk.BooleanVar(value=False)
        tk.Checkbutton(pop, text="Probability Distributions (.csv)", variable=self.b_opt_prob_csv).place(x=16, y=205)

        self.b_opt_prob_fig = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Probability Plots (.png)", variable=self.b_opt_prob_fig).place(x=16, y=230)

        self.b_opt_vis_csv = tk.BooleanVar(value=False)
        tk.Checkbutton(pop, text="3D Micelle Coordinates (.csv)", variable=self.b_opt_vis_csv).place(x=16, y=255)

        self.b_opt_vis_fig = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="3D Micelle Plot (.png)", variable=self.b_opt_vis_fig).place(x=16, y=280)

        tk.Label(pop, text="Batch Processing Mode:").place(x=16, y=305)
        self.batch_mode_var = tk.StringVar(value="click")
        tk.Radiobutton(pop, text="Auto-export individually as I click '>'", variable=self.batch_mode_var, value="click").place(x=16, y=325)
        tk.Radiobutton(pop, text="Process and export ALL files immediately", variable=self.batch_mode_var, value="all").place(x=16, y=345)

        def _start_batch():
            out_dir = self.batch_dir_var.get()
            if not os.path.exists(out_dir):
                try:
                    os.makedirs(out_dir)
                except Exception:
                    pass
            
            self.auto_export_config = {
                'dir': out_dir,
                'log': self.b_opt_log.get(),
                'raw': self.b_opt_raw.get(),
                'sim_csv': self.b_opt_sim_csv.get(),
                'sim_fig': self.b_opt_sim_fig.get(),
                'prob_csv': self.b_opt_prob_csv.get(),
                'prob_fig': self.b_opt_prob_fig.get(),
                'vis_csv': self.b_opt_vis_csv.get(),
                'vis_fig': self.b_opt_vis_fig.get()
            }
            
            self.current_batch_mode = self.batch_mode_var.get()
            pop.destroy()
            
            if self.current_batch_mode == "all":
                self.update_status("Starting batch process for all files... Please wait.")
                self.parent.after(100, self._ProcessBatchFile, 0, self.max_count)
            else:
                self._Forward() 

        def _skip_batch():
            self.auto_export_config = None
            pop.destroy()
            self._Forward() 

        tk.Button(pop, text="Start Batch Setup", command=_start_batch).place(height=40, width=380, x=16, y=375)
        tk.Button(pop, text="Skip Auto-Export", command=_skip_batch).place(height=30, width=380, x=16, y=425)
        pop.protocol("WM_DELETE_WINDOW", _skip_batch)

        return None

    def _ProcessBatchFile(self, index: int, total: int) -> None:
        """Asynchronously processes one file, then queues the next to prevent GUI freezing."""
        if index < total:
            self.count = index
            filenameshort = self.dir_list[self.count]
            self.file_path = os.path.join(self.source_path, filenameshort)
            self.origin = self.source_path  
            
            self.label_count.config(text=f'{self.count + 1}/{total}')
            
            try:
                self._PrepareFile() 
            except Exception as e:
                print(f"Error processing {filenameshort}: {e}")
                self.update_status(f"Error on {filenameshort}, skipping...")
            
            self.parent.after(50, self._ProcessBatchFile, index + 1, total)
        else:
            self.update_status(f"Successfully processed batch of {total} files!")
            self.auto_export_config = None
            
            self.button_forward.config(state=tk.DISABLED, bg="Light grey")
            if self.max_count > 1:
                self.button_backward.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

        return None

    def _Export(self, *args, **kwargs) -> None:
        """
        Single file manual export popup.
        """
        self.update_status("Opening export options...")

        pop = tk.Toplevel(self.parent)
        pop.title("Export Options")
        pop.geometry("420x400")
        pop.resizable(False, False)
        pop.geometry(f"+{self.parent.winfo_rootx() + 50}+{self.parent.winfo_rooty() + 50}")

        tk.Label(pop, text="Save Destination:").place(x=16, y=16)
        
        self.export_dir_var = tk.StringVar(value=self.last_opened_dir)
        entry_dir = tk.Entry(pop, textvariable=self.export_dir_var, state='readonly')
        entry_dir.place(x=16, y=40, width=300)
        
        def _browse_dir():
            chosen_dir = filedialog.askdirectory(initialdir=self.export_dir_var.get())
            if chosen_dir:  
                self.export_dir_var.set(chosen_dir)
                
        tk.Button(pop, text="Browse", command=_browse_dir).place(height=26, width=70, x=325, y=38)

        # --- EXPORT OPTIONS ---
        tk.Label(pop, text="Select files to generate and save:").place(x=16, y=80)

        self.exp_opt_log = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Append to Parameter Log (Record.csv)", variable=self.exp_opt_log).place(x=16, y=105)

        self.exp_opt_raw = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Original SAXS Data", variable=self.exp_opt_raw).place(x=16, y=130)

        self.exp_opt_sim_csv = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Simulated SAXS Profile (.csv)", variable=self.exp_opt_sim_csv).place(x=16, y=155)

        self.exp_opt_sim_fig = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="SAXS Profile Plot (.png)", variable=self.exp_opt_sim_fig).place(x=16, y=180)

        self.exp_opt_prob_csv = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Probability Distributions (.csv)", variable=self.exp_opt_prob_csv).place(x=16, y=205)

        self.exp_opt_prob_fig = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="Probability Plots (.png)", variable=self.exp_opt_prob_fig).place(x=16, y=230)

        self.exp_opt_vis_csv = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="3D Micelle Coordinates (.csv)", variable=self.exp_opt_vis_csv).place(x=16, y=255)

        self.exp_opt_vis_fig = tk.BooleanVar(value=True)
        tk.Checkbutton(pop, text="3D Micelle Plot (.png)", variable=self.exp_opt_vis_fig).place(x=16, y=280)

        btn_export = tk.Button(pop, text="Export Now", command=lambda: self._ExecuteExport(pop))
        btn_export.place(height=40, width=380, x=16, y=320)

        return None

    def _ExecuteExport(self, pop_window) -> None:
        save_dir = self.export_dir_var.get()
        if not save_dir or not os.path.exists(save_dir):
            self.update_status("Export cancelled: Invalid directory.")
            return

        self.update_status(f"Exporting to {os.path.basename(save_dir)}...")
        name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        opts = {
            'log': self.exp_opt_log.get(),
            'raw': self.exp_opt_raw.get(),
            'sim_csv': self.exp_opt_sim_csv.get(),
            'sim_fig': self.exp_opt_sim_fig.get(),
            'prob_csv': self.exp_opt_prob_csv.get(),
            'prob_fig': self.exp_opt_prob_fig.get(),
            'vis_csv': self.exp_opt_vis_csv.get(),
            'vis_fig': self.exp_opt_vis_fig.get()
        }
        
        self._SaveData(save_dir, name, opts)
        pop_window.destroy()
        self.update_status("Data exported successfully.")
        return None

    def _RunAutoExport(self, force_update=False) -> None:
        """Triggered automatically after Fit if a folder batch is setup"""
        if not getattr(self, 'auto_export_config', None): 
            return
        
        save_dir = self.auto_export_config['dir']
        if not os.path.exists(save_dir): 
            return
        
        name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        self._SaveData(save_dir, name, self.auto_export_config)
        
        if getattr(self, 'current_batch_mode', 'click') != "all" or force_update:
            self.update_status(f"Auto-exported {name} successfully.")
            
        return None

    def _ForceUpdateExport(self) -> None:
        """Triggered by the Update Export button to recalculate and overwrite batch files"""
        if getattr(self, 'auto_export_config', None) is not None:
            self.update_status("Applying changes and updating export...")
            try:
                self._UpdateParamsFromEntries()
            except ValueError:
                self.update_status("Error: Missing parameters in the boxes.")
                return None
                
            self._Simulate()
            self._Draw_sim()
            self._Draw_probability()
            
            self._RunAutoExport(force_update=True)
            self.update_status("Exported files successfully overwritten with new parameters!")
        else:
            self.update_status("Error: No active batch export to update.")
        return None

    def _SaveData(self, save_dir, name, opts) -> None:
        """Shared saving engine for single and batch exports"""
        try:
            if opts.get('log'):
                log_path = os.path.join(save_dir, "Record.csv")
                if not os.path.exists(log_path):
                    with open(log_path, 'w') as f:
                        f.write("CWD,Origin,File,Shape,p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7,m_0,m_1,m_2,s_0,s_1,s_2,Error,Rg_ML,Rg_GN,Comment\n")
                with open(log_path, 'a') as f:
                    origin = getattr(self, 'origin', 'Unknown')
                    file_path = getattr(self, 'file_path', 'Unknown')
                    shape = getattr(self, 'shape', 'Unknown')
                    p_0 = getattr(self, 'p_0', 0.0)
                    p_1 = getattr(self, 'p_1', 0.0)
                    p_2 = getattr(self, 'p_2', 0.0)
                    p_3 = getattr(self, 'p_3', 0.0)
                    p_4 = getattr(self, 'p_4', 0.0)
                    p_5 = getattr(self, 'p_5', 0.0)
                    p_6 = getattr(self, 'p_6', 0.0)
                    p_7 = getattr(self, 'p_7', 0.0)
                    m_0 = getattr(self, 'm_0', 0.0)
                    m_1 = getattr(self, 'm_1', 0.0)
                    m_2 = getattr(self, 'm_2', 0.0)
                    s_0 = getattr(self, 's_0', 0.0)
                    s_1 = getattr(self, 's_1', 0.0)
                    s_2 = getattr(self, 's_2', 0.0)
                    error = getattr(self, 'error', 0.0)
                    r_g_0 = getattr(self, 'r_g_0', 0.0)
                    r_g_1 = getattr(self, 'r_g_1', 0.0)
                    comment = self.entry_comment.get() if hasattr(self, 'entry_comment') else ""
                    
                    f.write(f'{self.cwd},{origin},{file_path},{shape},{p_0},{p_1},{p_2},{p_3},{p_4},{p_5},{p_6},{p_7},{m_0},{m_1},{m_2},{s_0},{s_1},{s_2},{error},{r_g_0},{r_g_1},{comment}\n')

            if opts.get('raw'):
                target = os.path.join(save_dir, f"{name}_raw" + os.path.splitext(self.file_path)[1])
                shutil.copy(self.file_path, target)

            if opts.get('sim_csv') and hasattr(self, 'I_sim'):
                sim_path = os.path.join(save_dir, f"{name}_simulated.csv")
                np.savetxt(sim_path, np.column_stack((self.q_arr, self.I_sim)), delimiter=",", header="q,I_sim", comments='')

            if opts.get('sim_fig') and hasattr(self, 'figure_s'):
                self.figure_s.savefig(os.path.join(save_dir, f"{name}_saxs_plot.png"), bbox_inches='tight')

            if opts.get('prob_csv') or opts.get('prob_fig'):
                self._Probability()  
                if hasattr(self, 'prob_0'):
                    params = self._CurrentParams()
                    raw_grid = np.linspace(0, 2, 257)[:-1]
                    probs = (self.prob_0, self.prob_1, self.prob_2)
                    means = getattr(self, 'm_0', 0.0), getattr(self, 'm_1', 0.0), getattr(self, 'm_2', 0.0)
                    sigmas = getattr(self, 's_0', 0.0), getattr(self, 's_1', 0.0), getattr(self, 's_2', 0.0)
                    values = getattr(self, 'p_0', 0.0), getattr(self, 'p_1', 0.0), getattr(self, 'p_2', 0.0)
                    
                    spec = getattr(self, "shape_spec", None)
                    if spec:
                        for i, (display, mean, sigma, value, probability) in enumerate(zip(spec.parameter_displays[:3], means, sigmas, values, probs)):
                            if display.model_to_plot is None:
                                continue
                            
                            x_values = display.model_to_plot(raw_grid, params)
                            clean_label = display.label.replace(" ", "_")
                            
                            if opts.get('prob_csv'):
                                csv_path = os.path.join(save_dir, f"{name}_prob_{clean_label}.csv")
                                np.savetxt(csv_path, np.column_stack((x_values, probability)), delimiter=",", header="value,probability", comments='')

                            if opts.get('prob_fig'):
                                fig = Figure(figsize=(4, 3), dpi=100)
                                ax = fig.add_subplot(111)
                                
                                ci_raw = np.array((mean - 1.96 * sigma, mean + 1.96 * sigma))
                                ci_values = display.model_to_plot(ci_raw, params)
                                current = value * display.entry_scale
                                
                                ax.plot(x_values, probability, color="blue")
                                ax.axvline(current, color="red")
                                ax.axvline(ci_values[0], color="black", linestyle="dashed")
                                ax.axvline(ci_values[1], color="black", linestyle="dashed")
                                ax.set_title(display.probability_title or f"{display.label} Probability")
                                ax.set_xlabel(display.probability_xlabel or display.label)
                                ax.set_ylabel(r"Probability Density")
                                if display.log_x: 
                                    ax.set_xscale("log")
                                ax.grid()
                                
                                fig.savefig(os.path.join(save_dir, f"{name}_prob_{clean_label}.png"), bbox_inches='tight')

            if (opts.get('vis_csv') or opts.get('vis_fig')) and hasattr(self, 's'):
                try:
                    scatterer_result = self.s.generate_scatterers(n=4096)
                    scatterers = scatterer_result[0] if isinstance(scatterer_result, tuple) else scatterer_result
                    scatterers = np.asarray(scatterers, dtype=float) * getattr(self, 'p_0', 1.0)
                    xs, ys, zs = scatterers[:, 0], scatterers[:, 1], scatterers[:, 2]
                    
                    if opts.get('vis_csv'):
                        vis_path = os.path.join(save_dir, f"{name}_3D_micelle.csv")
                        np.savetxt(vis_path, scatterers, delimiter=",", header="x,y,z", comments='')

                    if opts.get('vis_fig'):
                        fig = Figure(figsize=(5, 4), dpi=100)
                        ax = fig.add_subplot(111, projection="3d")
                        ax.scatter(xs, ys, zs, s=2)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_zlabel('Z')
                        ax.set_aspect('equal', adjustable='box')
                        fig.tight_layout()
                        fig.savefig(os.path.join(save_dir, f"{name}_3D_micelle.png"), bbox_inches='tight')

                except Exception as e:
                    print(f"Could not export 3D micelle: {e}")
        except Exception as e:
            print(f"Export failed: {e}")
            self.update_status(f"Export Error: {str(e)}")
            
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
                self.entry_0.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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

                spec = SHAPES_BY_CLASS.get(self._class)
                if spec is not None:
                    display = spec.parameter_displays[1]
                    physical_delta = delta / display.entry_scale
                    if display.unit == "angstrom":
                        physical_delta *= 10
                    self.p_1 += physical_delta
                    self.entry_1.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
                    self.entry_1.delete(0, tk.END)
                    self.entry_1.insert(0, f'{self.p_1 * display.entry_scale:.3f}')

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
                self.entry_2.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
                self.entry_3.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
                self.entry_4.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
                self.entry_5.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
                self.entry_6.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
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
                self.entry_7.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
                self.entry_7.delete(0, tk.END)
                self.entry_7.insert(0, f'{self.p_7:.3f}')

            case _:
                pass

        return None

    def _Classify(self, *args, **kwargs) -> None:

        self._Prepare()
        
        # 1. Run the standard AI pipeline
        result = classify_profile_hierarchical(
            I_arr=self.I_arr,
            q_log_arr=self.q_log_arr,
            qr_model=self.model_qr,
            family_classifier=self.model_family_cl,
            subclassifiers=self.model_subclassifiers,
        )

        best_spec = result.spec
        best_subclass_id = result.subclass_id

        # 2. BRUTE FORCE OVERRIDE: If the user checked the box, bypass the AI subclass guess
        if getattr(self, 'brute_force_var', None) and self.brute_force_var.get():
            self.update_status("Evaluating all subclasses (this may take a moment)...")
            self.parent.update()
            
            best_error = float('inf')
            
            # Identify all shapes that belong to the predicted main family
            family_specs = [
                s for s in SHAPES.values() 
                if s.family_key == result.family_key and s.model_key is not None
            ]
            
            for spec in family_specs:
                try:
                    # Predict parameters for this specific shape
                    shape_models = self.models["shape_models"][spec.model_key]
                    pred_0 = shape_models["radius"].predict(self.X, verbose=0)
                    pred_1 = shape_models["shape"].predict(self.X, verbose=0)
                    pred_2 = shape_models["pdi"].predict(self.X, verbose=0)
                    
                    m_0, s_0 = float(pred_0[0, 0]), float(pred_0[0, 1])
                    m_1, s_1 = float(pred_1[0, 0]), float(pred_1[0, 1])
                    m_2, s_2 = float(pred_2[0, 0]), float(pred_2[0, 1])
                    
                    if shape_models["rg"] is None:
                        m_3 = float(result.qr)
                    else:
                        pred_3 = shape_models["rg"].predict(self.X, verbose=0)
                        m_3 = float(pred_3[0, 0])
                        
                    params = spec.translate(m_0, s_0, m_1, s_1, m_2, s_2, m_3)
                    
                    # Run the mathematical simulation to see how well it actually fits
                    method = spec.scattering_class
                    sim_kwargs = spec.simulation_kwargs(params)
                    s = method(**sim_kwargs)
                    I_sim = s.Debye_scattering(q_arr=self.q_arr)
                    
                    # Calculate true error
                    error = mean_log_squared_error(self.I_arr, I_sim)
                    
                    # Keep the shape that produces the tightest fit
                    if error < best_error:
                        best_error = error
                        best_spec = spec
                        best_subclass_id = spec.subclass_id
                        
                except Exception as e:
                    print(f"Skipping {spec.display_name} in brute force due to error: {e}")
                    continue

        self.classification_result = result
        self.qr = result.qr
        self.family_key = result.family_key
        self.family_class_id = result.family_class_id
        self._family_class = result.family_class_id
        self.subclass_id = best_subclass_id

        family_score = 100 * result.family_score
        self._SetEntryText(self.entry_class_0, f'{100 - family_score:.3f}%', disabled=True)
        self._SetEntryText(self.entry_class_1, f'{family_score:.3f}%', disabled=True)

        self._ApplyShapeSpec(best_spec)
        self.family_class_id = result.family_class_id

        self._Reconfigure()

        return None

    def _Reconfigure(self, *args, **kwargs) -> None:

        self._EnableInputs()

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is not None:
            if hasattr(self, "parameter_labels"):
                for label, display in zip(self.parameter_labels, spec.parameter_displays):
                    label.config(text=f"{display.label}:")

            if hasattr(self, "parameter_aux_labels"):
                for index, label in self.parameter_aux_labels.items():
                    if index < len(spec.parameter_displays):
                        unit = "A" if spec.parameter_displays[index].unit == "angstrom" else spec.parameter_displays[index].unit
                        label.config(text=unit)

            unit = "A" if spec.primary_unit == "angstrom" else spec.primary_unit
            self.label_1.config(text=spec.primary_label)
            self.label_aux1.config(text=unit)
            if hasattr(self, "label_1_m") and hasattr(self, "label_1_s"):
                self.label_1_m.config(text=f"Mean ({unit}):")
                self.label_1_s.config(text=f"STD ({unit}):")

        return None

    def _EnableInputs(self, *args, **kwargs) -> None:

        self.entry_0.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_1.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_2.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_3.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_4.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_5.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_6.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.entry_7.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

        self.button_0_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_0_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_0_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_0_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_1_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_1_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_1_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_1_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_2_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_2_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_2_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_2_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_3_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_3_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_3_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_3_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_4_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_4_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_4_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_4_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_5_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_5_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_5_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_5_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_6_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_6_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_6_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_6_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_7_P_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_7_N_L.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_7_P_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.button_7_N_S.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

        return None

    def _ToggleFeatures(self, *args, **kwargs) -> None:

        if self.started:
            self.button_export.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_simulate.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_simulate.config(text='Simulate')
            self.button_visualize.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_probability.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_subclass_probability.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            if hasattr(self, 'button_update_export'):
                self.button_update_export.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        else:
            self.button_export.config(state=tk.DISABLED, bg="Light grey")
            self.button_simulate.config(state=tk.DISABLED, bg="Light grey")
            self.button_simulate.config(text='Start')
            self.button_visualize.config(state=tk.DISABLED, bg="Light grey")
            self.button_probability.config(state=tk.DISABLED, bg="Light grey")
            self.button_subclass_probability.config(state=tk.DISABLED, bg="Light grey")
            if hasattr(self, 'button_update_export'):
                self.button_update_export.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _Fit(self, *args, **kwargs) -> None:

        spec = SHAPES_BY_CLASS.get(self._class)
        
        if spec is None or getattr(spec, 'model_key', None) is None:
            shape_name = spec.display_name if spec else "this shape"
            self.update_status(f"Notice: No AI model for '{shape_name}'. Enter parameters manually.")
            
            self.drop_methods.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.drop_families.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_clear.config(state=tk.NORMAL)
            self.button_simulate.config(state=tk.NORMAL, text='Start')
            self._EnableInputs()
            
            if getattr(self, 'auto_export_config', None) is not None:
                self._RunAutoExport()
            return None

        self.shape_spec = spec

        self._GetPrediction()
        
        self._DisplayParams()
        self._DisplayProbability()

        self._Simulate()
        if hasattr(self, "I_sim"):
            self._Draw_sim()

        if self.fitted:
            self.started = True
            self.drop_methods.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.drop_families.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")

            self.button_export.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_visualize.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_probability.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_subclass_probability.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_simulate.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.button_simulate.config(text='Simulate')

            self._EnableInputs()

        if getattr(self, 'auto_export_config', None) is not None:
            self._RunAutoExport()

        return None
    
    def _GetPrediction(self, *args, **kwargs) -> None:

        try:
            shape_models = self.models["shape_models"][self.shape_spec.model_key]
            
            pred_0 = shape_models["radius"].predict(self.X, verbose=0)
            pred_1 = shape_models["shape"].predict(self.X, verbose=0)
            pred_2 = shape_models["pdi"].predict(self.X, verbose=0)
            
            self.m_0, self.s_0 = float(pred_0[0, 0]), float(pred_0[0, 1])
            self.m_1, self.s_1 = float(pred_1[0, 0]), float(pred_1[0, 1])
            self.m_2, self.s_2 = float(pred_2[0, 0]), float(pred_2[0, 1])
            
            if shape_models["rg"] is None:
                self.m_3 = float(self.qr)
            else:
                pred_3 = shape_models["rg"].predict(self.X, verbose=0)
                self.m_3 = float(pred_3[0, 0])

            self.params = self.shape_spec.translate(
                self.m_0, self.s_0, self.m_1, self.s_1, self.m_2, self.s_2, self.m_3
            )

            self.p_0 = self.params.p_0
            self.p_1 = self.params.p_1
            self.p_2 = self.params.p_2
            self.p_3 = self.params.p_3
            self.p_4 = self.params.p_4
            self.p_5 = self.params.p_5
            self.p_6 = self.params.p_6
            self.p_7 = self.params.p_7
            
            self.STD_0 = self.params.std_0
            self.STD_1 = self.params.std_1
            self.STD_2 = self.params.std_2
            self.r_g_0 = self.params.r_g_0
            
            try:
                self.Guinier_fit()
            except Exception as e:
                print(f"Guinier fit failed: {e}")
                self.r_g_1 = 0.0

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

        return None

    def _CurrentParams(self) -> PredictedParameters:

        return PredictedParameters(
            p_0=self.p_0,
            p_1=self.p_1,
            p_2=self.p_2,
            p_3=self.p_3,
            p_4=self.p_4,
            p_5=self.p_5,
            p_6=self.p_6,
            p_7=self.p_7,
            std_0=self.STD_0,
            std_1=self.STD_1,
            std_2=self.STD_2,
            r_g_0=self.r_g_0,
        )

    def _ParameterEntries(self) -> tuple[Entry, ...]:

        return (self.entry_0, self.entry_1, self.entry_2, self.entry_3, self.entry_4, self.entry_5, self.entry_6, self.entry_7)

    def _ProbabilityEntries(self) -> tuple[tuple[Entry, Entry, Entry], ...]:

        names = (
            "entry_0_m", "entry_0_s", "entry_0_d",
            "entry_1_m", "entry_1_s", "entry_1_d",
            "entry_2_m", "entry_2_s", "entry_2_d",
        )
        if not all(hasattr(self, name) for name in names):
            return ()

        return (
            (self.entry_0_m, self.entry_0_s, self.entry_0_d),
            (self.entry_1_m, self.entry_1_s, self.entry_1_d),
            (self.entry_2_m, self.entry_2_s, self.entry_2_d),
        )
        
    def _SetEntryText(self, entry: Entry, value: float | str, disabled: bool = False) -> None:

        entry.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        entry.delete(0, tk.END)
        entry.insert(0, value if isinstance(value, str) else f"{value:.3f}")
        if disabled:
            entry.config(state=tk.DISABLED, bg="Light grey")

        return None

    def _DisplayParams(self, *args, **kwargs) -> None:

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is None:
            return None

        param_values = (self.p_0, self.p_1, self.p_2, self.p_3, self.p_4, self.p_5, self.p_6, self.p_7)
        for entry, value, display in zip(self._ParameterEntries(), param_values, spec.parameter_displays):
            self._SetEntryText(entry, value * display.entry_scale)

        self.fitted = True

        self._SetEntryText(self.entry_8, self.r_g_0, disabled=True)
        self._SetEntryText(self.entry_9, self.r_g_1, disabled=True)

        return None

    def _DisplayProbability(self, *args, **kwargs) -> None:

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is None or not self._ProbabilityEntries():
            return None

        values = (self.p_0, self.p_1, self.p_2)
        stds = (self.STD_0, self.STD_1, self.STD_2)
        for entries, value, std, display in zip(self._ProbabilityEntries(), values, stds, spec.parameter_displays[:3]):
            mean_entry, std_entry, dev_entry = entries
            self._SetEntryText(mean_entry, value * display.entry_scale, disabled=True)
            self._SetEntryText(std_entry, std * display.entry_scale, disabled=True)
            self._SetEntryText(dev_entry, "0", disabled=True)

        return None

    def _Simulate(self, *args, **kwargs) -> None:

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is None or spec.scattering_class is None or spec.simulation_kwargs is None:
            return None

        params = self._CurrentParams()

        self.shape_spec = spec
        self.method = spec.scattering_class
        self.s = self.method(**spec.simulation_kwargs(params))
        
        self.I_sim = self.s.Debye_scattering(q_arr=self.q_arr)

        self._Error()

        return None
    
    def _Error(self, *args, **kwargs) -> None:

        error = mean_log_squared_error(self.I_arr, self.I_sim)
        self.error = error

        mmsle = error * 1000
        if mmsle > 0.5:
            bg = "#ffd6d6"
            fg = "red"
        elif mmsle < 0.05:
            bg = "#d8f5d0"
            fg = "green"
        else:
            bg = "Light grey"
            fg = "black"

        self.Entry_MSLE.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        self.Entry_MSLE.delete(0, tk.END)
        self.Entry_MSLE.insert(0, f'{mmsle:.3f}')
        self.Entry_MSLE.config(state=tk.DISABLED, disabledbackground=bg, disabledforeground=fg)

        return None

    def _Probability(self, *args, **kwargs) -> None:

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is not None and spec.deviance is not None:
            params = self._CurrentParams()
            self.dev_0, self.dev_1, self.dev_2 = spec.deviance(
                params,
                self.m_0,
                self.s_0,
                self.m_1,
                self.s_1,
                self.m_2,
                self.s_2,
            )

        temp = np.linspace(0, 2, 257)[:-1]

        prob_0 = np.exp(-np.square((temp - self.m_0) / (2 * self.s_0))) / self.s_0
        prob_1 = np.exp(-np.square((temp - self.m_1) / (2 * self.s_1))) / self.s_1
        prob_2 = np.exp(-np.square((temp - self.m_2) / (2 * self.s_2))) / self.s_2

        self.prob_0 = prob_0 / np.sqrt(2 * np.pi)
        self.prob_1 = prob_1 / np.sqrt(2 * np.pi)
        self.prob_2 = prob_2 / np.sqrt(2 * np.pi)

        if hasattr(self, 'entry_0_d') and self.entry_0_d.winfo_exists():
            self.entry_0_d.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.entry_0_d.delete(0, tk.END)
            self.entry_0_d.insert(0, f'{self.dev_0:.3f}')
            self.entry_0_d.config(state=tk.DISABLED, bg="Light grey")

            self.entry_1_d.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.entry_1_d.delete(0, tk.END)
            self.entry_1_d.insert(0, f'{self.dev_1:.3f}')
            self.entry_1_d.config(state=tk.DISABLED, bg="Light grey")

            self.entry_2_d.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
            self.entry_2_d.delete(0, tk.END)
            self.entry_2_d.insert(0, f'{self.dev_2:.3f}')
            self.entry_2_d.config(state=tk.DISABLED, bg="Light grey")

        return None
        
    def _Draw_qI(self, *args, **kwargs) -> None:

        filenameshort = os.path.basename(self.file_path)

        plot_s = self.plot_s
        canvas_s = self.canvas_s

        plot_s.clear()
        plot_s.plot(self.q_arr, self.I_arr, label='True')
        plot_s.set_title(filenameshort)
        plot_s.set_xlabel(r'q ($\AA^{-1}$)')
        plot_s.set_ylabel("Normalized Intensity")
        plot_s.set_xscale('log')
        plot_s.set_yscale('log')
        plot_s.legend()
        plot_s.grid()
        plot_s.figure.tight_layout() 
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
        plot_s.set_xlabel(r'q ($\AA^{-1}$)')
        plot_s.set_ylabel("Normalized Intensity")
        plot_s.set_xscale('log')
        plot_s.set_yscale('log')
        plot_s.legend()
        plot_s.grid()
        plot_s.figure.tight_layout() 
        canvas_s.draw()

        return None

    def _Draw_probability(self, *args, **kwargs) -> None:

        if not self._ProbabilityEntries():
            return None

        self._Probability()

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is None:
            return None

        params = self._CurrentParams()
        raw_grid = np.linspace(0, 2, 257)[:-1]
        means = getattr(self, 'm_0', 0.0), getattr(self, 'm_1', 0.0), getattr(self, 'm_2', 0.0)
        sigmas = getattr(self, 's_0', 0.0), getattr(self, 's_1', 0.0), getattr(self, 's_2', 0.0)
        values = getattr(self, 'p_0', 0.0), getattr(self, 'p_1', 0.0), getattr(self, 'p_2', 0.0)
        probs = getattr(self, 'prob_0', [0]*256), getattr(self, 'prob_1', [0]*256), getattr(self, 'prob_2', [0]*256)
        plots = (self.plot_0, self.plot_1, self.plot_2)
        canvases = (self.canvas_0, self.canvas_1, self.canvas_2)

        for plot, canvas, display, mean, sigma, value, probability in zip(plots, canvases, spec.parameter_displays[:3], means, sigmas, values, probs):
            if display.model_to_plot is None:
                continue

            x_values = display.model_to_plot(raw_grid, params)
            ci_raw = np.array((mean - 1.96 * sigma, mean + 1.96 * sigma))
            ci_values = display.model_to_plot(ci_raw, params)
            current = value * display.entry_scale

            plot.clear()
            plot.plot(x_values, probability, color="blue")
            plot.axvline(current, color="red")
            plot.axvline(ci_values[0], color="black", linestyle="dashed")
            plot.axvline(ci_values[1], color="black", linestyle="dashed")
            plot.set_title(display.probability_title or f"{display.label} Probability Function")
            plot.set_xlabel(display.probability_xlabel or display.label)
            plot.set_ylabel(r"Probability Density")
            if display.log_x:
                plot.set_xscale("log")
            else:
                plot.set_xscale("linear")
            plot.grid()
            plot.figure.tight_layout() 
            canvas.draw()

        return None

    def _Draw_probability_0(self, *args, **kwargs) -> None:

        self._Draw_probability()

        return None

    def _Draw_probability_1(self, *args, **kwargs) -> None:

        self._Draw_probability()

        return None


    def _UpdateParamsFromEntries(self) -> None:

        spec = SHAPES_BY_CLASS.get(self._class)
        if spec is None:
            return None

        param_names = ("p_0", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7")
        for entry, name, display in zip(self._ParameterEntries(), param_names, spec.parameter_displays):
            setattr(self, name, float(entry.get()) / display.entry_scale)

        return None

    def _Simulate_as(self, *args, **kwargs) -> None:
        self.update_status("Starting simulation...")
        
        self.update_idletasks() 

        try:
            self._UpdateParamsFromEntries()
        except ValueError:
            self.update_status("Error: Missing parameters in the boxes.")
            print("Cannot simulate: Missing parameters.")
            return None

        if not self.started:
            if self.fitted:
                self.started = True
                self._ToggleFeatures()

        self._Simulate()
        self._Draw_sim()
        self._Draw_probability()

        self.update_status("Simulation completed.")

        return None

    def _Simulate_as_0(self, *args, **kwargs) -> None:

        self._Simulate_as()

        return None

    def _Simulate_as_1(self, *args, **kwargs) -> None:

        self._Simulate_as()

        return None


    def _Visualize(self, *args, **kwargs) -> None:
        self.update_status("Visualizing results...")
        n = 4096
        s = self.s

        scatterer_result = s.generate_scatterers(n=n)
        if isinstance(scatterer_result, tuple):
            scatterers = scatterer_result[0]
        else:
            scatterers = scatterer_result
        scatterers = np.asarray(scatterers, dtype=float)
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

        fig.tight_layout() 
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

        if 'button' in kwargs:
            _button = kwargs['button']
            _button.configure(text=filenameshort)

        _, self.I_arr = load_saxs_profile(file_path, self.q_arr)

        return None

    def _Prepare(self, *args, **kwargs) -> None:

        self.X = prepare_model_input(self.I_arr)

        return None

    def _PrepareFile(self, *args, **kwargs) -> None:
        """Safely loads, prepares, and fits the file. Prevents silent crashing on ML failure."""
        self._Clear()
        self.button_simulate.config(state=tk.NORMAL, background="SystemButtonFace", foreground="black")
        
        try:
            self.get_qI()
            self._Draw_qI()
            self._Classify()
            self._Fit()
            self.update_status("File loaded successfully.")
        except Exception as e:
            error_msg = str(e)
            print(f"Error during file preparation: {error_msg}")
            self.update_status(f"Prediction Error: {error_msg}")

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

        self.Y = interpolate_for_classifier(self.q_log_arr, self.I_arr, self.qr)

        return None

    def Guinier_fit(self, *args, **kwargs) -> None:

        self.r_g_1 = guinier_radius(
            q_arr       =self.q_arr,
            I_arr       =self.I_arr,
            radius_guess=self.r_g_0,
            class_id    =self._family_class,
        )

        return None


def main(*args, **kwargs) -> int:
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

    return 0


if __name__ == '__main__':
    # --- MAC PACKAGING MULTIPROCESSING FIX ---
    multiprocessing.freeze_support()
    # -----------------------------------------
    main()