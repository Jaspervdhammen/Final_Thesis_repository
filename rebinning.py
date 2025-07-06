import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import Forwardmodel as v  # External module for spectrum modeling
from functions import sigma_filter, pandexo
from petitRADTRANS.planet import Planet


class SpectrumProcessor:
    """
    A class to handle reading, processing, and rebinding of spectroscopic data.
    """

    def __init__(self, resolution_factor=20, crop_range=(3.71, 3.84), rebin_threshold=0.01):
        self.resolution_factor = resolution_factor
        self.crop_range = crop_range
        self.rebin_threshold = rebin_threshold
        self.colors = ["#c99b38", "#eddca5", "#00b0be", "#8fd7d7"]
        self._configure_plotting()

    def _configure_plotting(self):
        """Configure Matplotlib settings for plots."""
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.dpi': 150
        })

    @staticmethod
    def read_csv_data(filename):
        """
        Load and filter 4-column CSV data based on w ≤ 1.

        Parameters:
            filename (str): Path to the CSV file.

        Returns:
            tuple: Arrays of x, y, z, and w values.
        """
        data = np.loadtxt(filename, delimiter=' ')
        data = data[data[:, 3] <= 1]
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    @staticmethod
    def read_csv_data2(filename):
        """
        Load 2-column CSV data.

        Parameters:
            filename (str): Path to the CSV file.

        Returns:
            tuple: Arrays of x and y values.
        """
        data = np.loadtxt(filename, delimiter=None)
        return data[:, 0], data[:, 1]

    def calculate_real_spectrum(self, R):
        """
        Build and return a synthetic transmission spectrum using a forward model.

        Parameters:
            R (int): Spectral resolution.

        Returns:
            tuple: Wavelengths and transmission values of the synthetic spectrum.
        """
        model = v.V1298SpectrumModel(
            mass=25 * 5.97e27, temp=800, wolk=None,
            H2O=10**-1.64, CO2=10**-2.65, CO=10**-2.05,
            SO2=10**-4.79, CH4=10**-5.38, R=R,
            data_path="/home/jappie/pythoncode/base codes/V1298_Tau_b_data"
        )
        model.build_model()
        actual_x, actual_y = model.calculate_spectrum()
        actual_y **= 2  # Convert to relative squared depth
        actual_y *= 1e6

        xl, xr = self.crop_range
        mask = (actual_x >= xl) & (actual_x <= xr)
        return actual_x[mask], actual_y[mask]

    @staticmethod
    def calculate_resolution(x):
        """
        Estimate the spectral resolution R from wavelength array.

        Parameters:
            x (array): Wavelengths.

        Returns:
            int: Estimated resolution.
        """
        delta_lambda = np.diff(x)
        R = x[:-1] / delta_lambda
        return int(np.ceil(np.median(R) / 10.0)) * 10

    @staticmethod
    def rebinning(factor, x, y, error):
        """
        Rebin spectral data to a lower resolution.

        Parameters:
            factor (int): Rebinning factor.
            x, y, error (arrays): Spectral wavelength, transmission, and error.

        Returns:
            tuple: Rebinned x, y, and error arrays.
        """
        n_bins = len(x) // factor
        rebinned_x, rebinned_y, rebinned_error = [], [], []

        for i in range(n_bins):
            start = i * factor
            end = start + factor
            bin_x = x[start:end]
            bin_y = y[start:end]
            bin_error = error[start:end]

            rebinned_x.append(np.average(bin_x))
            rebinned_y.append(np.median(bin_y))
            rebinned_error.append(np.sqrt(np.sum(bin_error**2)) / len(bin_error))

        if n_bins * factor < len(x):
            bin_x = x[n_bins * factor:]
            bin_y = y[n_bins * factor:]
            bin_error = error[n_bins * factor:]
            rebinned_x.append(np.average(bin_x))
            rebinned_y.append(np.median(bin_y))
            rebinned_error.append(np.sqrt(np.sum(bin_error**2)) / len(bin_error))

        return np.array(rebinned_x), np.array(rebinned_y), np.array(rebinned_error)

    @staticmethod
    def not_too_close(x, y, err, threshold):
        """
        Filter out data points that are too close in wavelength.

        Parameters:
            x, y, err (arrays): Input data arrays.
            threshold (float): Minimum wavelength spacing.

        Returns:
            tuple: Filtered arrays.
        """
        mask = np.where(np.abs(np.diff(x)) > threshold)[0]
        return x[mask], y[mask], err[mask]

    def process_and_save(self):
        """
        Run the main processing routine and write rebinned data to file.
        """
        plt.figure(figsize=(10, 6))

        pandexowave, pandexotransit, pandexoerror = pandexo(50, 800, "Y")
        pandexowaverebin, pandexotransitrebin, pandexoerrorrebin = self.rebinning(
            self.resolution_factor, pandexowave, pandexotransit, pandexoerror
        )

        # Avoid log(0) or undefined bins
        pandexobin = [pandexowaverebin[0] / 2 if i == 0 else val for i, val in enumerate(pandexowaverebin)]

        print(f"len(pandexowave): {len(pandexowave)}")
        print(f"len(pandexoerror): {len(pandexoerror)}")
        print(f"len(pandexotransit): {len(pandexotransit)}")

        V1298 = Planet.get('V1298 tau b')
        resolution = self.calculate_resolution(pandexowaverebin)
        print(f"Resolution: {resolution}")

        with open('output_TROUBLESHOOTING.txt', 'w') as file:
            for x, b, y, e in zip(pandexowaverebin, pandexobin, pandexotransitrebin, pandexoerrorrebin):
                file.write(f'{x} {b} {y} {e} \n')


# ----------------------
# Run if main
# ----------------------

if __name__ == "__main__":
    processor = SpectrumProcessor()
    processor.process_and_save()
