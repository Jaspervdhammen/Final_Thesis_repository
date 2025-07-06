import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Forwardmodel import V1298SpectrumModel as VModel  # Adjust import if needed
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functions2 import replace_line_at_index
import matplotlib.patches as patches
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import psutil
import os
colors = ["#c99b38","#eddca5","#00b0be","#8fd7d7"]

def parse_z_column(z_raw):
    """Convert string column to floats, handling 'None' as np.nan."""
    return np.array([float(z) if z.lower() != 'none' else 0 for z in z_raw])


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # in bytes
    mem_mb = mem_bytes / (1024 ** 2)
    print(f"Memory usage: {mem_mb:.2f} MB")


# Function to calculate the height of the CO2 peak
def calculate_height_CO2_peak(wave, transitdepth):
    wave = np.array(wave)
    transitdepth = np.array(transitdepth)

    minimum_mask = np.where((wave > 3) & (wave < 4))
    transit_minimum = np.min(transitdepth[minimum_mask])

    maximum_mask = np.where(wave > 4)
    transit_maximum = np.max(transitdepth[maximum_mask])

    return transit_maximum - transit_minimum


# Function to load data and normalize it
def load_data(filename):
    data = np.loadtxt(filename)
    wave, transitdepth, error = data[:, 0], data[:, 2], data[:, 3]
    transitdepth -= np.min(transitdepth[np.where((wave > 3) & (wave < 4))])  # Normalize all to minimum = 0
    return wave, transitdepth, error

def load_data2(filename):
    data = np.loadtxt(filename)
    wave, transitdepth, error = data[:, 0], data[:, 1], data[:, 2]
    transitdepth -= np.min(transitdepth[np.where((wave > 3) & (wave < 4))])  # Normalize all to minimum = 0
    return wave, transitdepth, error



# Cloud type to filename tag
cloud_options = {
    'None': 'CNM',
    '1e-3': 'CYM',
    '5e-3': 'CYYM'
}


class SpectrumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("V1298 Tau b Spectrum Explorer")

        # Main parameter inputs
        self.mass_var = tk.DoubleVar(value=50)
        self.temp_var = tk.DoubleVar(value=800)
        self.wolk_var = tk.StringVar(value='None')
        self.res_var = tk.IntVar(value=1000)
        self.cloud_gui = tk.StringVar(value='None')
        self.mass_gui = tk.StringVar(value='50')  # File mass (not physical mass)

        # Molecular abundances (log10 values)
        self.h2o_var = tk.StringVar(value='-1.64')
        self.ch4_var = tk.StringVar(value='-5.39')
        self.co2_var = tk.StringVar(value='-2.65')
        self.co_var = tk.StringVar(value='-2.05')
        self.so2_var = tk.StringVar(value='-4.79')

                # Somewhere in your __init__ or GUI setup
        self.z_selector = tk.StringVar()
        self.z_dropdown = ttk.Combobox(root, textvariable=self.z_selector, state="readonly")
        self.z_dropdown.bind("<<ComboboxSelected>>", self.plot_progress)
        self.z_dropdown.pack()  # Or use grid/place

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.z_selector = tk.StringVar()
        self.z_dropdown = ttk.Combobox(root, textvariable=self.z_selector, state="readonly")
        self.z_dropdown.bind("<<ComboboxSelected>>", self.plot_progress)
        self.z_dropdown.pack()

        self.build_controls()
        self.build_plot()
        self.im = None  # Initialize im attribute here
        self.cbar = None

        # Reset button
        self.reset_button = tk.Button(root, text="Reset Plot", command=self.reset_plot)
        self.reset_button.pack()

    def reset_plot(self):
        # Remove all axes from the figure
        self.fig.clf()

        # Create a new single axis
        self.ax = self.fig.add_subplot(111)

        # Plot empty data and set a placeholder title
        self.ax.plot([], [])
        self.ax.set_title("Empty Plot")

        # Redraw the canvas to reflect changes
        self.canvas.draw()

    
    def build_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # Use plot_frame here
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def build_controls(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Physical parameters
        tk.Label(frame, text="Mass (Earth Masses)").pack()
        tk.Entry(frame, textvariable=self.mass_var).pack()

        tk.Label(frame, text="Temperature (K)").pack()
        tk.Entry(frame, textvariable=self.temp_var).pack()

        tk.Label(frame, text="Cloud Top Pressure ('None', 1e-3, 5e-3)").pack()
        tk.Entry(frame, textvariable=self.wolk_var).pack()

        tk.Label(frame, text="Resolution").pack()
        tk.Entry(frame, textvariable=self.res_var).pack()

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Molecular abundances (log10 inputs)
        tk.Label(frame, text="H₂O (log10 abundance)").pack()
        tk.Entry(frame, textvariable=self.h2o_var).pack()

        tk.Label(frame, text="CH₄ (log10 abundance)").pack()
        tk.Entry(frame, textvariable=self.ch4_var).pack()

        tk.Label(frame, text="CO₂ (log10 abundance)").pack()
        tk.Entry(frame, textvariable=self.co2_var).pack()

        tk.Label(frame, text="CO (log10 abundance)").pack()
        tk.Entry(frame, textvariable=self.co_var).pack()

        tk.Label(frame, text="SO₂ (log10 abundance)").pack()
        tk.Entry(frame, textvariable=self.so2_var).pack()

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Observed file controls
        tk.Label(frame, text="Observed File Mass (e.g., 10, 25, 50)").pack()
        tk.OptionMenu(frame, self.mass_gui, "10", "25", "50").pack()

        tk.Label(frame, text="Observed File Cloud Type").pack()
        tk.OptionMenu(frame, self.cloud_gui, *cloud_options.keys()).pack()

        tk.Button(frame, text="Plot Model + Observed", command=self.plot_model_and_data).pack(pady=5)
        tk.Button(frame, text="Plot Observed Only", command=self.plot_data).pack(pady=5)
        # New button for progress plot
        tk.Button(frame, text="Progress", command=self.plot_progress).pack(pady=5)
        tk.Button(frame, text="Run Batch", command=self.run_batch).pack(pady=5)

    def build_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def plot_data(self):
        
        self.ax.clear()
        mass = self.mass_gui.get()
        cloud = self.cloud_gui.get()
        cloud_tag = cloud_options[cloud]
        filename = f"output{cloud_tag}{mass}.txt"

        try:
            wave, transit, error = load_data(filename)
            height = calculate_height_CO2_peak(wave, transit)

            self.ax.errorbar(wave, transit, yerr=error, fmt='o', label="Observed", color='black')
            self.ax.set_xlim(2.87, 5.2)
            self.ax.set_xlabel("Wavelength (μm)")
            self.ax.set_ylabel("Normalized Transit Depth")
            self.ax.set_title(f"Observed Spectrum (Mass = {mass}, Clouds = {cloud})")
            self.ax.text(0.05, 0.95, f"CO₂ Peak Height: {height:.5f}",
                         transform=self.ax.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            self.ax.legend()
            self.canvas.draw()
        except FileNotFoundError:
            print(f"File not found: {filename}")

    def plot_model_and_data(self):
        # return
        self.ax.clear()  # Clear the plot first

        mass = self.mass_gui.get()
        cloud = self.cloud_gui.get()
        cloud_tag = cloud_options[cloud]
        filename = f"output{cloud_tag}{mass}.txt"
        # filename = "outputCYM50_rebin.txt"

        try:
            wave_obs, transit_obs, error_obs = load_data(filename)
            height = calculate_height_CO2_peak(wave_obs, transit_obs)
            
            # Plot observed data
            self.ax.errorbar(wave_obs, transit_obs, yerr=error_obs, fmt='o', label="Observed", color=colors[2])
            self.ax.set_xlim(2.87, 5.2)
            self.ax.set_xlabel("Wavelength (μm)")
            self.ax.set_ylabel("Shifted Transit Depth in ppm")
            self.ax.set_title(f"Forward Model + data of planet")
            self.ax.text(0.05, 0.95, f"CO₂ Peak Height: {height:.5f}",
                        transform=self.ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            # Calculate and plot the model
            mass_kg = self.mass_var.get() * 5.97e27
            temp = self.temp_var.get()
            res = self.res_var.get()
            wolk_input = self.wolk_var.get()
            wolk = None if wolk_input.lower() == 'none' else float(wolk_input)

            abundances = {
                'h2o': 10 ** float(self.h2o_var.get()),
                'ch4': 10 ** float(self.ch4_var.get()),
                'co2': 10 ** float(self.co2_var.get()),
                'co': 10 ** float(self.co_var.get()),
                'so2': 10 ** float(self.so2_var.get())
            }

            model = VModel(mass=mass_kg, temp=temp, wolk=wolk, R=res,
                        H2O=abundances['h2o'],
                        CH4=abundances['ch4'],
                        CO2=abundances['co2'],
                        CO=abundances['co'],
                        SO2=abundances['so2'])

            model.build_model()
            wave_model, transit_model = model.calculate_spectrum()
            mask = np.where((wave_model > 2.71) & (wave_model < 5.17))
            transit_model = transit_model ** 2

            # Interpolate to observed wavelengths
            model_interp_raw = np.interp(wave_obs, wave_model, transit_model)
            model_interp_raw -= np.max(model_interp_raw)
            # Initial shift = difference between mean data and model
            shift = np.mean(transit_obs - model_interp_raw)
            delta = np.average(error_obs)

            min_chi2 = np.inf
            best_shift = shift
            increasing_counter = 0

            while increasing_counter < 3:
                # Apply shift
                shifted_model = model_interp_raw + shift
                residuals = transit_obs - shifted_model
                chi_squared = np.sum((residuals / error_obs) ** 2)
                dof = len(transit_obs) - 5
                reduced_chi_squared = chi_squared / dof

                if reduced_chi_squared < min_chi2:
                    min_chi2 = reduced_chi_squared
                    best_shift = shift
                    increasing_counter = 0
                else:
                    increasing_counter += 1

                shift += delta  # Increment shift

            
            # Final model with best shift
            best_model = model_interp_raw + best_shift
            self.ax.plot(wave_obs, best_model, label="Model", color=colors[0])

            self.ax.text(0.05, 0.88, f"Reduced $\chi^2$: {min_chi2:.4f}",
                        transform=self.ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            self.ax.legend()
            self.canvas.draw()

        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error generating model: {e}")

        
    def run_batch(self):
    #     # Clear or create the output file first

        # Get fixed parameters
        res = self.res_var.get()
        wolk_input = self.wolk_var.get()
        cloud = self.cloud_gui.get()
        cloud_tag = cloud_options[cloud]

        abundances = {
            'h2o': 10 ** float(self.h2o_var.get()),
            'ch4': 10 ** float(self.ch4_var.get()),
            'co2': 10 ** float(self.co2_var.get()),
            'co': 10 ** float(self.co_var.get()),
            'so2': 10 ** float(self.so2_var.get())
        }
        

        # with tqdm(total = int(((140-60)/10) * (1700/100) * 1) , desc="Running batch...", unit="run") as pbar:
        for file in ["CNM50.txt","CYM50.txt","CYYM50.txt","CNM25.txt","CYM25.txt","CYYM25.txt","CNM10.txt","CYM10.txt","CYYM10.txt" ]:
                filename = file  # You can adjust the output file if needed
                with open(filename, "w") as f:
                    f.write("# mass(g) temperature(K) reduced_chi_squared\n")

                # obs_filename = f"output{filename}"
                obs_filename = f"output{filename}"
                wave_obs, transit_obs, error_obs = load_data(obs_filename) 


                if filename == "CNM50.txt" or "CYM50.txt" or "CYYM50.txt":
                    for clouds in [None,1e-3,5e-3,7.5e-3,2.5e-3]:
                        for mass_val in range(10,150, 10):  # 10 to 100 step 10
                            prev_chi2 = np.inf
                            skip_rest = False  # Flag to skip remaining temperatures if 3 bad chi2s exceed 10
                            outer_increading_counter = 0
                            for temp_val in range(300, 2001, 100):  # 300 to 2000 step 100
                                print_memory_usage()
                                # pbar.update(1)
                                

                                if skip_rest:
                                    with open(filename, "a") as f:
                                        f.write(f"{mass_val * 5.97e27} {temp_val} {clouds} 50\n")
                                    continue

                                else:
                                    mass_kg = mass_val * 5.97e27

                                    model = VModel(mass=mass_kg, temp=temp_val, wolk=clouds, R=1000,
                                                H2O=abundances['h2o'],
                                                CH4=abundances['ch4'],
                                                CO2=abundances['co2'],
                                                CO=abundances['co'],
                                                SO2=abundances['so2'])
                                    model.build_model()
                                    wave_model, transit_model = model.calculate_spectrum()

                                    mask = np.where((wave_model > 2.71) & (wave_model < 5.17))
                                    transit_model = transit_model ** 2

                                    model_interp_raw = np.interp(wave_obs, wave_model, transit_model)
                                    model_interp_raw -= np.max(model_interp_raw)

                                    shift = np.mean(transit_obs - model_interp_raw)
                                    delta = np.average(error_obs)

                                    min_chi2 = np.inf
                                    best_shift = shift
                                    increasing_counter = 0

                                    while increasing_counter < 10:
                                        shifted_model = model_interp_raw + shift
                                        residuals = transit_obs - shifted_model
                                        chi_squared = np.sum((residuals / error_obs) ** 2)
                                        dof = len(transit_obs) - 5
                                        reduced_chi_squared = chi_squared / dof

                                        if reduced_chi_squared < min_chi2:
                                            min_chi2 = reduced_chi_squared
                                            best_shift = shift
                                            increasing_counter = 0
                                        else:
                                            increasing_counter += 1

                                        shift += delta
                                    
                                    if min_chi2 > prev_chi2:
                                        outer_increading_counter += 1
                                    else:
                                        outer_increading_counter = 0

                                    # If chi² is bad 3x in a row and high, stop for this mass
                                    if outer_increading_counter > 3 and min_chi2 > 50:
                                        skip_rest = False

                                    with open(filename, "a") as f:
                                        f.write(f"{mass_kg} {temp_val} {clouds} {min_chi2}\n")

                                    print(f"Done mass={mass_val}, temp={temp_val}, clouds = {clouds} chi²={min_chi2:.4f}")
                                    prev_chi2 = min_chi2
                                    import gc
                                    del model, wave_model, transit_model, model_interp_raw, shifted_model, residuals
                                    del mask, shift, delta, chi_squared, dof, reduced_chi_squared
                                    gc.collect()

                                # except FileNotFoundError:
                                #     print(f"Observed file not found for mass {mass_val}: {obs_filename}")
                                # except Exception as e:
                                #     print(f"Error for mass={mass_val}, temp={temp_val}: {e}")
            
                # data = np.genfromtxt(filename, dtype=None, encoding=None, delimiter=None, names=True)

                if filename == "CYYM25.txt" or filename =="CNM25.txt" or filename == "CYM25.txt": # or "CNM50.txt":
                    for clouds in [None,1e-3,5e-3,2.5e-3,7.5e-3]:
                        for mass_val in range(5, 55, 5):  # 10 to 100 step 10
                            prev_chi2 = np.inf
                            skip_rest = False  # Flag to skip remaining temperatures if 3 bad chi2s exceed 10
                            outer_increading_counter = 0
                            for temp_val in range(300, 2001, 100):  # 300 to 2000 step 100
                                print_memory_usage()
                                # pbar.update(1)
                                

                                if skip_rest:
                                    with open(filename, "a") as f:
                                        f.write(f"{mass_val * 5.97e27} {temp_val} {clouds} 50\n")
                                    continue

                                try:
                                    mass_kg = mass_val * 5.97e27

                                    model = VModel(mass=mass_kg, temp=temp_val, wolk=clouds, R=res,
                                                H2O=abundances['h2o'],
                                                CH4=abundances['ch4'],
                                                CO2=abundances['co2'],
                                                CO=abundances['co'],
                                                SO2=abundances['so2'])
                                    model.build_model()
                                    wave_model, transit_model = model.calculate_spectrum()

                                    mask = np.where((wave_model > 2.71) & (wave_model < 5.17))
                                    transit_model = transit_model ** 2

                                    model_interp_raw = np.interp(wave_obs, wave_model, transit_model)
                                    model_interp_raw -= np.max(model_interp_raw)

                                    shift = np.mean(transit_obs - model_interp_raw)
                                    delta = np.average(error_obs)

                                    min_chi2 = np.inf
                                    best_shift = shift
                                    increasing_counter = 0

                                    while increasing_counter < 3:
                                        shifted_model = model_interp_raw + shift
                                        residuals = transit_obs - shifted_model
                                        chi_squared = np.sum((residuals / error_obs) ** 2)
                                        dof = len(transit_obs) - 5
                                        reduced_chi_squared = chi_squared / dof

                                        if reduced_chi_squared < min_chi2:
                                            min_chi2 = reduced_chi_squared
                                            best_shift = shift
                                            increasing_counter = 0
                                        else:
                                            increasing_counter += 1

                                        shift += delta
                                    
                                    if min_chi2 > prev_chi2:
                                        outer_increading_counter += 1
                                    else:
                                        outer_increading_counter = 0

                                    # If chi² is bad 3x in a row and high, stop for this mass
                                    if outer_increading_counter > 3 and min_chi2 > 50:
                                        skip_rest = True

                                    with open(filename, "a") as f:
                                        f.write(f"{mass_kg} {temp_val} {clouds} {min_chi2}\n")

                                    print(f"Done mass={mass_val}, temp={temp_val}, clouds = {clouds} chi²={min_chi2:.4f}")
                                    prev_chi2 = min_chi2
                                    import gc
                                    del model, wave_model, transit_model, model_interp_raw, shifted_model, residuals
                                    del mask, shift, delta, chi_squared, dof, reduced_chi_squared
                                    gc.collect()

                                except FileNotFoundError:
                                    print(f"Observed file not found for mass {mass_val}: {obs_filename}")
                                except Exception as e:
                                    print(f"Error for mass={mass_val}, temp={temp_val}: {e}")

                if filename == "CYYM10.txt" or filename =="CNM10.txt" or filename == "CYM10.txt": # or "CNM50.txt":
                    for clouds in [None,1e-3,5e-3,2.5e-3,7.5e-3]:
                        for mass_val in range(2, 22, 2):  # 10 to 100 step 10
                            prev_chi2 = np.inf
                            skip_rest = False  # Flag to skip remaining temperatures if 3 bad chi2s exceed 10
                            outer_increading_counter = 0
                            for temp_val in range(300, 2001, 100):  # 300 to 2000 step 100
                                print_memory_usage()
                                # pbar.update(1)
                                

                                if skip_rest:
                                    with open(filename, "a") as f:
                                        f.write(f"{mass_val * 5.97e27} {temp_val} {clouds} 50\n")
                                    continue

                              
                                mass_kg = mass_val * 5.97e27

                                model = VModel(mass=mass_kg, temp=temp_val, wolk=clouds, R=res,
                                                H2O=abundances['h2o'],
                                                CH4=abundances['ch4'],
                                                CO2=abundances['co2'],
                                                CO=abundances['co'],
                                                SO2=abundances['so2'])
                                model.build_model()
                                wave_model, transit_model = model.calculate_spectrum()

                                mask = np.where((wave_model > 2.71) & (wave_model < 5.17))
                                transit_model = transit_model ** 2

                                model_interp_raw = np.interp(wave_obs, wave_model, transit_model)
                                model_interp_raw -= np.max(model_interp_raw)

                                shift = np.mean(transit_obs - model_interp_raw)
                                delta = np.average(error_obs)

                                min_chi2 = np.inf
                                best_shift = shift
                                increasing_counter = 0

                                while increasing_counter < 3:
                                    shifted_model = model_interp_raw + shift
                                    residuals = transit_obs - shifted_model
                                    chi_squared = np.sum((residuals / error_obs) ** 2)
                                    dof = len(transit_obs) - 5
                                    reduced_chi_squared = chi_squared / dof

                                    if reduced_chi_squared < min_chi2:
                                        min_chi2 = reduced_chi_squared
                                        best_shift = shift
                                        increasing_counter = 0
                                    else:
                                        increasing_counter += 1

                                    shift += delta
                                
                                if min_chi2 > prev_chi2:
                                    outer_increading_counter += 1
                                else:
                                    outer_increading_counter = 0

                                # If chi² is bad 3x in a row and high, stop for this mass
                                if outer_increading_counter > 3 and min_chi2 > 50:
                                    skip_rest = True

                                with open(filename, "a") as f:
                                    f.write(f"{mass_kg} {temp_val} {clouds} {min_chi2}\n")

                                print(f"Done mass={mass_val}, temp={temp_val}, clouds = {clouds} chi²={min_chi2:.4f}")
                                prev_chi2 = min_chi2
                                import gc
                                del model, wave_model, transit_model, model_interp_raw, shifted_model, residuals
                                del mask, shift, delta, chi_squared, dof, reduced_chi_squared
                                gc.collect()

    #                             except FileNotFoundError:
    #                                 print(f"Observed file not found for mass {mass_val}: {obs_filename}")
    #                             except Exception as e:
    #                                 print(f"Error for mass={mass_val}, temp={temp_val}: {e}")



    def plot_progress(self, event=None):
        from matplotlib.patches import Patch
        cloud_key = self.cloud_gui.get()
        mass_val = self.mass_gui.get()
        cloud_tag = cloud_options[cloud_key]
        # filename = f"{cloud_tag}{mass_val}.txt"
        filename = "CYM50_rebin.txt"
        legend_patches = [Patch(facecolor=colors[0], edgecolor='black', label=r"$\sigma < 1$"),Patch(facecolor=colors[1], edgecolor='black', label=r"$1 < \sigma < 2$"),Patch(facecolor=colors[2], edgecolor='black', label=r"$2 < \sigma < 3$"),]


        try:
            data = np.genfromtxt(filename, dtype=None, encoding=None, delimiter=None, names=True)

            x = data["massg"]
            y = data["temperatureK"]
            z = data["clouds"]
            w = data["reduced_chi_squared"]
            minimal_chi2 = np.min(w)

            z_unique = np.unique(z)
            z_unique = parse_z_column(z_unique)

            if not self.z_dropdown["values"]:
                self.z_dropdown["values"] = [f"{val}" for val in z_unique]
                self.z_selector.set(f"{z_unique[0]}")
                return

            selected_z = float(self.z_selector.get())
            # selected_z = 0

            mask = np.where(parse_z_column(z) == selected_z)
            x_sel = x[mask]
            y_sel = y[mask]
            w_sel = w[mask]


            if len(x_sel) == 0:
                print(f"[WARNING] No data for z = {selected_z}")
                return

            x_unique = np.unique(x_sel)
            y_unique = np.unique(y_sel)
            grid = np.full((len(y_unique), len(x_unique)), np.nan)
            grid_sqrt = np.full_like(grid, np.nan)

            for xi, yi, wi in zip(x_sel, y_sel, w_sel):
                ix = np.where(x_unique == xi)[0][0]
                iy = np.where(y_unique == yi)[0][0]
                if wi == 50:
                    wi = 50 + minimal_chi2
                if wi - minimal_chi2 == 0:
                    wi += 0.0001
                
                print(minimal_chi2)

                val = min(wi - minimal_chi2, 50)
                grid[iy, ix] = val
                grid_sqrt[iy, ix] = min(np.sqrt(val), 50)

            if hasattr(self, 'ax_dataset') and self.ax_dataset in self.fig.axes:
                self.fig.delaxes(self.ax_dataset)
                self.ax_dataset = None
            self.fig.clf()

            axs = self.fig.subplots(1, 1, sharey=True)
            # self.ax = axs[0]
            self.ax2 = axs

            def draw_heatmap(ax, data_grid, title, cmap='managua', show_values=True):
                im = ax.imshow(data_grid, origin='lower', vmin=np.nanmin(data_grid),
                            vmax=np.nanmax(data_grid), cmap=cmap, interpolation='none')

                ax.set_xticks(np.arange(len(x_unique)))
                ax.set_xticklabels([f"{round(v / 5.97e27, 0)}" for v in x_unique], rotation=45)
                ax.set_yticks(np.arange(len(y_unique)))
                ax.set_yticklabels([f"{round(v, 0)}" for v in y_unique])
                ax.set_xlabel("Mass (Earth Masses)")
                ax.set_ylabel("Temperature (K)")
                ax.set_title(title)

                if show_values:
                    for i in range(len(y_unique)):
                        for j in range(len(x_unique)):
                            val = data_grid[i, j]
                            if not np.isnan(val):
                                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=5)

                return im

            title1 = f"Cloud = Clear sky \n Standard Deviation ranges" if selected_z == 0 else f"Cloud = {round(selected_z * 1000, 1)} mbar \n Standard Deviation ranges"
            title2 = "Sigma ranges"

            im1 = draw_heatmap(self.ax2, grid_sqrt, title1)
            # im2 = draw_heatmap(self.ax2, grid_sqrt, title2)

                        # Draw contour boxes for Δχ² = 1, 4, 9

            
            thresholds = [1, 4, 9]
            color_map = {"low": colors[0], "mid": colors[1], "high": colors[2],"not": colors[3]}
            grid_to_use = grid

            for ax in [self.ax2]:
                for i in range(len(y_unique)):
                    for j in range(len(x_unique)):
                        val = grid_to_use[i, j]
                        if not np.isnan(val):
                            if val < thresholds[0]:
                                color = color_map["low"]
                            elif thresholds[0] <= val < thresholds[1]:
                                color = color_map["mid"]
                            elif thresholds[1] <= val < thresholds[2]:
                                color = color_map["high"]
                            else:
                                color = color_map["not"]  # outside target range
                            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                    linewidth=1, edgecolor=color, facecolor=color)
                            ax.add_patch(rect)

            # cbar1 = self.fig.colorbar(im1, ax=self.ax, label=r"$\Delta \chi^2$")
            # Add legend to the sqrt-scaled plot (self.ax2)
            self.ax2.legend(handles=legend_patches, title="Significance Zones", loc='upper right', fontsize=8, title_fontsize=9)
            # cbar2 = self.fig.colorbar(im2, ax=self.ax2, label=r"$\sqrt{\Delta \chi^2}$")
            
            self.fig.tight_layout()
            self.fig.savefig(f"MASS{mass_val}CLOUDS{cloud_key}_CL{selected_z}.png")
            self.canvas.draw_idle()


            # Optional: Remove colorbars afterwards if necessary
            # cbar1.remove()

        except FileNotFoundError:
            print(f"[ERROR] File not found: {filename}")

# Run the GUI
root = tk.Tk()

app = SpectrumApp(root)
root.mainloop()
