import os
import shutil

import tkinter as tk
from tkinter import Button, StringVar, filedialog, Entry, Label

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from datetime import datetime


class MainApplication(tk.Frame):
    # Constructor that initializes the app and sets up the interface
    def __init__(self, parent) -> None:
        self.parent = parent
        self._Setup()
        return None

    # Sets up the necessary components for the app
    def _Setup(self) -> None:
        self._SetNumbers()
        self._SetFolders()
        self._SetUI()
        return None
    
    # Configures the UI elements
    def _SetUI(self) -> None:
        parent = self.parent
        tk.Frame.__init__(self, parent)
        parent.title('Autosubtraction')
        parent.geometry("1080x400")
        parent.protocol("WM_DELETE_WINDOW", parent.quit())
        self.dy = 16    # Y-offset for placing UI elements
        self._SetButtons()
        self._SetPlots()
        return None
    
    # Initializes numeric arrays and default variables
    def _SetNumbers(self) -> None:
        q_log_arr = np.arange(-2.0, 0.0, np.true_divide(1, 128) - 2*np.log10(2))
        q_arr = np.power(10, q_log_arr)
        self.q_arr = q_arr
        self.q_crit = 0.2
        self.loaded_0 = False
        self.loaded_1 = False
        self.loaded_2 = False
        return None
    
    # Sets up the folder structure for file management
    def _SetFolders(self) -> None:
        cwd = os.getcwd()
        username = os.getlogin()
        current = datetime.now()
        current = current.strftime('%Y%m%d')
        base_path = os.path.join(cwd, 'Subtraction')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # Creates unique working directory using a counter
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
        img_dir= os.path.join(working_dir, 'Images')

        # Creates log file for records
        log_end = 'csv'
        log_file = f'Record.{log_end}'
        log_path = os.path.join(working_dir, log_file)

        # Creates directories
        os.makedirs(working_dir)
        os.makedirs(raw_dir)
        os.makedirs(back_dir)
        os.makedirs(sub_dir)
        with open(log_path, "a") as f:
            f.write("Raw,Background,Subtracted,Scale factor,Minimum q value,Comments\n")

        # Saves directory paths to instance variables
        self.cwd = cwd
        self.base_path = base_path
        self.working_dir = working_dir
        self.raw_dir = raw_dir
        self.back_dir = back_dir
        self.sub_dir = sub_dir
        self.img_dir = img_dir
        self.log_path = log_path
        return None
    
    # Sets up buttons for user interaction
    def _SetButtons(self) -> None:
        
        parent = self.parent
        dy = self.dy
        reg = parent.register(self._Callback)
        height = 3*dy - 2
        width = 320

        # Creates and places buttons with appropriate commands
        button_single = Button(parent, text="Single", command=self._Single)
        button_single.place(height=height, width=width/2 - 2, x=20, y=1*dy)
        button_single.config(state=tk.NORMAL)
        button_multiple = Button(parent, text="Multiple", command=self._Multiple)
        button_multiple.place(height=height, width=width/2 - 2, x=182, y=1*dy)
        button_multiple.config(state=tk.NORMAL)
        button_back = Button(parent, text="Buffer", command=self._Load_1)
        button_back.place(height=height, width=width, x=20, y=4*dy)
        button_back.config(state=tk.NORMAL)
        button_sub = Button(parent, text="Subtract", command=self._Subtract)
        button_sub.place(height=height, width=width, x=20, y=7*dy)
        button_sub.config(state=tk.DISABLED)    # Initially disabled

        # Navigation and utility buttons (all initially disabled)
        button_backward = Button(parent, text="<", command=self._Backward)
        button_backward.place(height=height, width=width/2 - 2, x=20, y=10*dy)
        button_backward.config(state=tk.DISABLED)
        button_forward = Button(parent, text=">", command=self._Forward)
        button_forward.place(height=height, width=width/2 - 2, x=182, y=10*dy)
        button_forward.config(state=tk.DISABLED)
        button_clear = Button(parent, text="Clear", command=self._Clear)
        button_clear.place(height=height, width=width/2 - 2, x=20, y=13*dy)
        button_clear.config(state=tk.DISABLED)
        button_export = Button(parent, text="Export", command=self._Export)
        button_export.place(height=height, width=width/2 - 2, x=182, y=13*dy)
        button_export.config(state=tk.DISABLED)
        button_cutoff = Button(parent, text="Cutoff", command=self._Cutoff)
        button_cutoff.place(height=height, width=120, x=20, y=16*dy)

        # Entry fields for q value and comments
        var_q = StringVar()
        entry_q = Entry(parent, textvariable=var_q)
        entry_q.place(height=height, width=100, x=140, y=16*dy)
        entry_q.config(validate="key", validatecommand=(reg, '%P'))
        entry_q.insert(0, f'{self.q_crit}')
        label_count = Label(parent, text='NA')
        label_count.place(height=height, width=100, x=240, y=16*dy)
        label_comment = Label(parent, text='Comments')
        label_comment.place(height=30, width=width, x=20, y=19*dy)
        var_comment = StringVar()
        entry_comment = Entry(parent, textvariable=var_comment)
        entry_comment.place(height=height, width=width, x=20, y=21*dy)

        # Saves widgets as instance variables
        self.button_single = button_single
        self.button_multiple = button_multiple
        self.button_back = button_back
        self.button_sub = button_sub
        self.button_forward = button_forward
        self.button_backward = button_backward
        self.button_clear = button_clear
        self.button_export = button_export
        self.var_q = var_q
        self.var_comment = var_comment
        self.label_count = label_count
        return None
    
    # Creates plots for data visualization
    def _SetPlots(self) -> None:
        parent = self.parent
        figure_0 = Figure(figsize=(4, 4), dpi=64)
        plot_0 = figure_0.add_subplot(1, 1, 1)
        plot_0.set_title("Raw")
        plot_0.set_xlabel(r'q ($\AA^{-1}$)')
        plot_0.set_ylabel('Scattering Intensity')
        plot_0.set_xscale('log')
        plot_0.set_yscale('log')
        canvas_0 = FigureCanvasTkAgg(figure_0, parent)
        canvas_0.get_tk_widget().place(height=360, width=360, x=360, y=20)
        
        figure_1 = Figure(figsize=(4, 4), dpi=64)
        plot_1 = figure_1.add_subplot(1, 1, 1)
        plot_1.set_title("Subtracted")
        plot_1.set_xlabel(r'q ($\AA^{-1}$)')
        plot_1.set_ylabel('Scattering Intensity')
        plot_1.set_xscale('log')
        plot_1.set_yscale('log')
        canvas_1 = FigureCanvasTkAgg(figure_1, parent)
        canvas_1.get_tk_widget().place(height=360, width=360, x=720, y=20)
        
        self.figure_0 = figure_0
        self.figure_1 = figure_1
        self.plot_0 = plot_0
        self.plot_1 = plot_1
        self.canvas_0 = canvas_0
        self.canvas_1 = canvas_1
        return None
    
    # Validation callback for entry fields
    def _Callback(self, input_: str, *args, **kwargs) -> bool:
        try:
            if input_ == '':
                return True
            float(input_)
        except ValueError:
            return False
        return True
    
    # Clears data and resets UI elements
    def _Clear(self) -> None:
        self._ClearButtons()
        self._ClearVariables()
        self._ClearFile()
        return None
    
    # Resets button states
    def _ClearButtons(self) -> None:
        self.button_single.config(text='Single')
        self.button_multiple.config(text='Multiple')
        self.button_back.config(text='Buffer')
        self.button_sub.config(state=tk.DISABLED)
        self.button_clear.config(state=tk.DISABLED)
        self.button_export.config(state=tk.DISABLED)
        return None
    
    # Resets loaded flags
    def _ClearVariables(self) -> None:
        self.loaded_0 = False
        self.loaded_1 = False
        self.loaded_2 = False
        return None
    
    # Clears plots and resets titles
    def _ClearFile(self) -> None:
        self.plot_0.clear()
        self.plot_0.set_title("Raw")
        self.plot_0.set_xlabel(r'q ($\AA^{-1}$)')
        self.plot_0.set_ylabel('Scattering Intensity')
        self.plot_0.set_xscale('log')
        self.plot_0.set_yscale('log')
        self.plot_0.grid()

        self.canvas_0.draw()

        self.plot_1.clear()
        self.plot_1.set_title("Subtracted")
        self.plot_1.set_xlabel(r'q ($\AA^{-1}$)')
        self.plot_1.set_ylabel('Scattering Intensity')
        self.plot_1.set_xscale('log')
        self.plot_1.set_yscale('log')
        self.plot_1.grid()
        
        self.canvas_1.draw()
        
        return None
    
    # Saves data to files and exports plots
    def _Export(self) -> None:
        
        raw_dir = self.raw_dir
        back_dir = self.back_dir
        sub_dir = self.sub_dir
        
        log_path = self.log_path
        
        file_0 = self.file_0
        file_1 = self.file_1

        # Defines target paths for raw and back files
        target_0 = os.path.join(raw_dir, os.path.basename(file_0))
        target_1 = os.path.join(back_dir, os.path.basename(file_1))

        # Extracts file name without extension for output
        name = os.path.basename(file_0).split('.')[0]
        target_2 = os.path.join(sub_dir, f'{name}.csv')

        # Copies files to specified directories
        shutil.copy(file_0, target_0)
        shutil.copy(file_1, target_1)

        # Stacks and saves numerical data into a CSV file
        arr = np.vstack((self.qs_2, self.Is_2, self.ss_2)).T
        np.savetxt(target_2, arr, delimiter=",")

        # Appends export information to a log file
        comment = self.var_comment.get()
        with open(log_path, "a") as f:
            f.write(f"{target_0},{target_1},{target_2},{self.alpha},{self.q_crit},{comment}\n")

        # Saves plots as images
        figure_0 = self.figure_0
        figure_1 = self.figure_1
        img_dir = self.img_dir
        img_path_0 = os.path.join(img_dir, f'{name}_raw.csv')
        img_path_1 = os.path.join(img_dir, f'{name}_sub.csv')
        
        figure_0.savefig(img_path_0)
        figure_1.savefig(img_path_1)
        
        return None
    
    # Loads a file and prepares it for processing
    def _LoadFile(self) -> None:
        root = self.parent
        root.path = filedialog.askopenfilename(
            initialdir=os.getcwd(), 
            title="Select a File"
        )
        filename = root.path

        if filename:
            if self.working_index == 0:
                self.label_count.config(text='NA')
                self.button_forward.config(state=tk.DISABLED)
                self.button_backward.config(state=tk.DISABLED)
                self.single = True
            
            filenameshort = os.path.basename(filename)
            name_len = len(filenameshort)
            folder_name = filename[:-name_len]
            
            self.working_origin = folder_name
            self.working_file = filename
            self._PrepareFile()
        
        return None
    
    # Wrapper functions for loading specific files
    def _Load_0(self) -> None:
        
        self.working_index = 0
        self.loaded_0 = True
        self._LoadFile()
        
        return None
    
    
    def _Load_1(self) -> None:
        
        self.working_index = 1
        self.loaded_1 = True
        self._LoadFile()
        
        return None
    
    
    def _Single(self) -> None:
        self._Load_0()
        return None
    
    
    def _Multiple(self) -> None:
                
        filenames = filedialog.askopenfilenames(
            initialdir=os.getcwd(), 
            title="Select a File"
        )
                
        if filenames:
            
            self.filenames = filenames
            self.length = len(filenames)
            self.count = -1
            self.single = False
            
            self._Move(forward=True)
        
        return
    
    # Handles navigation between files
    def _Move(self, forward=bool) -> None:
        self.working_index = 0
        self.loaded_0 = True

        # Updates file navigation index
        if forward:
            self.count = min(self.length - 1, self.count + 1)
        else:
            self.count = max(0, self.count - 1)

        # Updates button states and labels based on position
        if self.count == self.length - 1:
            self.button_forward.config(state=tk.DISABLED)
            self.button_backward.config(state=tk.NORMAL)
        elif self.count == 0:
            self.button_forward.config(state=tk.NORMAL)
            self.button_backward.config(state=tk.DISABLED)
        else:
            self.button_forward.config(state=tk.NORMAL)
            self.button_backward.config(state=tk.NORMAL)
        
        self.label_count.config(text=f'{self.count + 1}/{self.length}')

        # Updates working file information
        filename = self.filenames[self.count]
        filenameshort = os.path.basename(filename)
        
        name_len = len(filenameshort)
        folder_name = filename[:-name_len]
        
        self.working_origin = folder_name
        self.working_file = filename
        
        self._PrepareFile()

        return None
    
    # Navigates forward in the file list
    def _Forward(self) -> None:
        self._Move(forward=True)
        return None
    
    # Navigates backward in the file list
    def _Backward(self) -> None:
        self._Move(forward=False)
        return None
    
    # Prepares a file for processing by updating its data
    def _PrepareFile(self) -> None:
        
        self.get_qI()
        self._UpdateData()
        self._UpdateButton()
        self._UpdatePlot_0()
        
        return None
    
    # Updates class variables with the current file's data
    def _UpdateData(self) -> None:
        if self.working_index == 0:
            self.qs_0 = self.working_qs
            self.Is_0 = self.working_Is
            self.ss_0 = self.working_ss
            self.origin_0 = self.working_origin
            self.file_0 = self.working_file
        elif self.working_index == 1:
            self.qs_1 = self.working_qs
            self.Is_1 = self.working_Is
            self.ss_1 = self.working_ss
            self.origin_1 = self.working_origin
            self.file_1 = self.working_file
        else:
            pass
        return None
    
    # Updates button states based on file load status
    def _UpdateButton(self) -> None:
                
        if self.working_index == 0:
            if self.single:
                self.button_single.config(text=os.path.basename(self.working_file))
                self.button_multiple.config(text='Multiple')
            else:
                self.button_single.config(text='Single')
                self.button_multiple.config(text=os.path.basename(self.working_file))
        elif self.working_index == 1:
            self.button_back.config(text=os.path.basename(self.working_file))
        else:
            pass
        
        if self.loaded_0 and self.loaded_1:
            self.button_sub.config(state=tk.NORMAL)
            self.button_clear.config(state=tk.NORMAL)
            self.button_export.config(state=tk.NORMAL)
        elif self.loaded_0 or self.loaded_1:
            self.button_sub.config(state=tk.DISABLED)
            self.button_clear.config(state=tk.NORMAL)
            self.button_export.config(state=tk.DISABLED)
        else:
            self.button_sub.config(state=tk.DISABLED)
            self.button_clear.config(state=tk.DISABLED)
            self.button_export.config(state=tk.DISABLED)
        
        return None
    
    # Updates the first plot with loaded data
    def _UpdatePlot_0(self) -> None:
        
        if self.loaded_0 and not self.loaded_1:
            self.plot_0.clear()
            self.plot_0.plot(self.qs_0, self.Is_0)
            self.plot_0.axvline(x=self.q_crit, color='r')
            self.plot_0.set_title("Raw")
            self.plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.plot_0.set_ylabel('Scattering Intensity')
            self.plot_0.set_xscale('log')
            self.plot_0.set_yscale('log')
            self.plot_0.grid()
            
            self.canvas_0.draw()
            
        elif self.loaded_1 and not self.loaded_0:
            self.plot_0.clear()
            self.plot_0.plot(self.qs_1, self.Is_1)
            self.plot_0.axvline(x=self.q_crit, color='r')
            self.plot_0.set_title("Raw")
            self.plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.plot_0.set_ylabel('Scattering Intensity')
            self.plot_0.set_xscale('log')
            self.plot_0.set_yscale('log')
            self.plot_0.grid()
            
            self.canvas_0.draw()

        elif self.loaded_0 and self.loaded_1:
            self.plot_0.clear()
            self.plot_0.plot(self.qs_0, self.Is_0, label='Raw')
            self.plot_0.plot(self.qs_1, self.Is_1, label='Background')
            self.plot_0.axvline(x=self.q_crit, color='r')
            self.plot_0.set_title("Raw")
            self.plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.plot_0.set_ylabel('Scattering Intensity')
            self.plot_0.set_xscale('log')
            self.plot_0.set_yscale('log')
            self.plot_0.legend()
            self.plot_0.grid()
            
            self.canvas_0.draw()
        
        else:
            self.plot_0.clear()
            self.plot_0.set_title("Raw")
            self.plot_0.set_xlabel(r'q ($\AA^{-1}$)')
            self.plot_0.set_ylabel('Scattering Intensity')
            self.plot_0.set_xscale('log')
            self.plot_0.set_yscale('log')
            self.plot_0.legend()
            self.plot_0.grid()
            
            self.canvas_0.draw()
        
        return None
    
    # Perform background subtraction and update the second plot
    def _Subtract(self) -> None:
        
        self._AutoSubtract()
        self._UpdatePlot_1()
        
        return None
    
    # Automatically compute the subtraction factor and perform subtraction
    def _AutoSubtract(self) -> None:
        
        q_crit = self.var_q.get()
        q_crit = float(q_crit)
        
        if self.loaded_0 and self.loaded_1:

            temp_0 = self.Is_0[self.qs_0 > q_crit]
            temp_1 = self.Is_1[self.qs_1 > q_crit]
            
            sum_01 = np.sum(temp_0*temp_1)
            sum_11 = np.sum(np.square(temp_1))
            
            alpha = sum_01/sum_11
            self.alpha = alpha
            
            self.qs_2 = self.qs_0
            self.Is_2 = self.Is_0 - alpha*self.Is_1
            self.ss_2 = np.sqrt(np.square(self.ss_0) + np.square(alpha*self.ss_1))
            self.loaded_2 = True
        
        return None
    
    # Update the second plot with subtracted data
    def _UpdatePlot_1(self) -> None:
        if self.loaded_2:
            self.plot_1.clear()
            self.plot_1.plot(self.qs_2, self.Is_2)
            self.plot_1.set_title("Subtracted")
            self.plot_1.set_xlabel(r'q ($\AA^{-1}$)')
            self.plot_1.set_ylabel('Scattering Intensity')
            self.plot_1.set_xscale('log')
            self.plot_1.set_yscale('log')
            self.plot_1.grid()
            
            self.canvas_1.draw()
        
        return None
    
    # Set q cutoff value and update plot
    def _Cutoff(self) -> None:
        
        q_crit = float(self.var_q.get())
        
        working_qs = self.working_qs
        
        if len(working_qs[working_qs >= q_crit]) < 8:
            q_crit = working_qs[-8]
        
        self.q_crit = q_crit
        self._UpdatePlot_0()
        
        return None

    # Reads scattering data from the working file and stores it in class attributes.
    # Supports files with two or three columns (q, I, [s])
    def get_qI(self, *args, **kwargs) -> None:
        # Get the working file path and extract the short filename and file extension
        working_file = self.working_file
        filenameshort = os.path.basename(working_file)
        end = filenameshort[-3:]

        # Update the button text if provided in kwargs
        if 'button' in kwargs:
            _button = kwargs['button']
            _button.configure(text=filenameshort)

        # Initialize a temporary list to store file data
        temp = list()

        # Open the working file for reading
        with open(working_file, 'r') as f:
            # Skip lines until a line starts with a digit (indicating data)
            while f.readline()[0] not in "0123456789":
                continue

            # Read and parse the file line by line
            while True:
                line = f.readline()
                # Determine the delimiter based on the file extension
                if line:
                    if end == 'csv':
                        num = len(line.split(','))
                    else:
                        num = len(line.split())

                    # Handle data with two or three columns
                    if num == 2:
                        q, I = line.split()
                        temp.append((float(q), float(I)))
                    else:
                        q, I, s = line.split()
                        temp.append((float(q), float(I), float(s)))
                else:
                    break

        # Convert the temporary list to a numpy array for easier manipulation
        temp = np.array(temp)

        # Extract columns for q-values and intensity (I-values)
        qs = temp[:, 0]
        Is = temp[:, 1]

        # Compute error (ss) based on the number of columns in the data
        if temp.shape[1] == 2:
            ss = np.sqrt(Is) # Assume error is the square root of intensity if not provided
        else:
            ss = temp[:, 2]

        # Store data into class attributes for further use
        self.working_qs = qs
        self.working_Is = Is  
        self.working_ss = ss
        
        return None


# Initializes and runs the main application loop
def main(*args, **kwargs) -> int:
    # Create the main Tkinter application window
    root = tk.Tk()

    #Initializes and packs the main application
    MainApplication(root).pack(side="top", fill="both", expand=True)

    # Starts the Tkinter event loop
    root.mainloop()
    
    return 0

# Runs the main function if this file is executed as a script
if __name__ == '__main__':
    main()
