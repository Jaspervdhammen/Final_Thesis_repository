# In this file, the forward model of V1298 Tau B can be created, plotted and saved
# Created by Jasper van der Hammen
# Date 5-7-2025
#-------------------------------------------------------------------------------------------------------------------------------


import os
import numpy as np
import matplotlib.pyplot as plt
import petitRADTRANS.physical_constants as cst
from petitRADTRANS.planet import Planet
from petitRADTRANS.spectral_model import SpectralModel
import functions as f


colors = ["#c99b38", "#eddca5", "#00b0be", "#8fd7d7"]

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150
})

mmm = 2.5028755  # Mean molecular mass?


def readoutdata(filename):
    """
    Reads two-column data from a text file, ignoring malformed lines.

    Parameters:
        filename (str): Path to the file to read.

    Returns:
        tuple: Two lists containing the first and second column values as strings.
    """
    column1 = []
    column2 = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                column1.append(parts[0])
                column2.append(parts[1])
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return column1, column2


def mass_fraction(volume, molecular_weight, mean_molecular_mass):
    """
    Calculate the mass fraction of a species.

    Parameters:
        volume (float): Volume or concentration parameter.
        molecular_weight (float): Molecular weight of the species.
        mean_molecular_mass (float): Mean molecular mass of the atmosphere.

    Returns:
        float: Mass fraction of the species.
    """
    return (volume * molecular_weight) / mean_molecular_mass


class V1298SpectrumModel:
    """
    Class to model the transmission spectrum of V1298 Tau b exoplanet.

    Attributes:
        mass (float): Planet mass in grams.
        temp (float): Atmospheric temperature in K.
        wolk (float or None): Cloud top pressure or None.
        CO2, H2O, SO2, CH4, CO (float): Mass fractions for gases.
        R (int): Spectral resolution.
        offset (float): Wavelength or flux offset.
        data_path (str): Path to observational data.
        spectral_model (SpectralModel): petitRADTRANS spectral model instance.
        wavelengths (np.ndarray): Wavelength array.
        transit_radii (np.ndarray): Transit radius array.
    """

    def __init__(self, mass, temp, wolk, CO2, H2O, SO2, CH4, CO, R, data_path="", offset=0.0):
        """
        Initialize the V1298SpectrumModel instance.

        Parameters:
            mass (float): Mass of the planet (grams).
            temp (float): Isothermal temperature (K).
            wolk (float or None): Cloud top pressure (bar) or None.
            CO2, H2O, SO2, CH4, CO (float): Imposed mass fractions of gases.
            R (int): Spectral resolution.
            data_path (str): Directory path to observational data.
            offset (float): Offset for plotting or data alignment.
        """
        self.V1298 = Planet.get('V1298 tau b')
        self.mass = mass
        self.temp = temp
        self.wolk = wolk
        self.CO2 = CO2
        self.H2O = H2O
        self.SO2 = SO2
        self.CH4 = CH4
        self.CO = CO
        self.R = R
        self.offset = offset
        self.data_path = data_path
        self.spectral_model = None
        self.wavelengths = None
        self.transit_radii = None
        self.filtered_wavelengths = None
        self.filtered_transit_radii = None

        print(f"Current mass: {self.mass / (5.97e27):.3f} Earth Mass")

    def build_model(self):
        """
        Build the petitRADTRANS spectral model using input parameters.

        Returns:
            SpectralModel: Configured petitRADTRANS spectral model instance.
        """
        planet_radius = self.V1298.radius
        self.spectral_model = SpectralModel(
            pressures=np.logspace(-8, 2, 100),
            line_species=[f"H2O.R{self.R}", f"CO2.R{self.R}", f"CH4.R{self.R}",
                          f"CO-NatAbund.R{self.R}", f"SO2.R{self.R}"],
            rayleigh_species=['H2', 'He'],
            gas_continuum_contributors=['H2-H2', 'H2-He'],
            wavelength_boundaries=[0.3, 15],
            planet_radius=planet_radius,
            reference_gravity=(6.674e-8) * self.mass / planet_radius ** 2,
            reference_pressure=1e-2,
            is_observed=True,
            is_around_star=True,
            system_distance=211.15820511 * cst.s_cst.light_year * 1e2,
            star_effective_temperature=self.V1298.star_effective_temperature,
            star_radius=self.V1298.star_radius,
            orbit_semi_major_axis=self.V1298.orbit_semi_major_axis,
            temperature_profile_mode="isothermal",
            temperature=self.temp,
            use_equilibrium_chemistry=False,
            opaque_cloud_top_pressure=self.wolk,
            imposed_mass_fractions={
                f"H2O.R{self.R}": self.H2O,
                f"CH4.R{self.R}": self.CH4,
                f"CO2.R{self.R}": self.CO2,
                f"CO-NatAbund.R{self.R}": self.CO,
                f"SO2.R{self.R}": self.SO2,
                f"H2S.R{self.R}": mass_fraction(10 ** -8.4, 34.082, mmm),
                f"HCN.R{self.R}": mass_fraction(10 ** -10, 27.0253, mmm),
                f"C2H2.R{self.R}": mass_fraction(10 ** -10.5, 26.038, mmm),
                f"NH3.R{self.R}": mass_fraction(10 ** -9.8, 17.031, mmm)
            },
            filling_species={"H2": 37, "He": 12, "Ne": 0.001}
        )
        return self.spectral_model

    def calculate_spectrum(self):
        """
        Calculate the transmission spectrum from the spectral model.

        Returns:
            tuple: Arrays of filtered wavelengths (microns) and normalized transit radii.
        """
        self.wavelengths, self.transit_radii = self.spectral_model.calculate_spectrum(mode='transmission')
        self.filtered_wavelengths = self.wavelengths[0] * 1e4  # Convert to microns
        self.filtered_transit_radii = self.transit_radii[0] / self.V1298.star_radius

        print(f"Spectrum points: wavelengths={len(self.filtered_wavelengths)}, transit radii={len(self.filtered_transit_radii)}")
        return self.filtered_wavelengths, self.filtered_transit_radii

    def plot_transmission(self):
        """
        Plot the transmission spectrum of the planet along with observational data.
        """
        self.data()
        key = list(self.data_.keys())[0]
        data_sub = self.data_[key].copy()

        X = data_sub['CENTRALWAVELNG']
        actual_x = self.wavelengths[0] * 1e4
        actual_y = (self.transit_radii[0] / self.V1298.star_radius) ** 2

        yr = []
        for xi in X:
            closest_x = actual_x[np.argmin(np.abs(actual_x - xi))]
            index_actual = np.where(actual_x == closest_x)[0][0]
            yr.append(actual_y[index_actual])
        yr = np.array(yr)

        offset = np.abs(np.mean(np.array(data_sub['PL_RATROR']) ** 2 - yr))
        print(f"Offset between data and model: {offset}")

        plt.plot(
            self.wavelengths[0] * 1e4,
            ((self.transit_radii[0] / self.V1298.star_radius) ** 2) * 1e6,
            label="Original model",
            color=colors[0]
        )
        plt.xscale("log")
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('$(R_p / R_s)^2$ in ppm')
        plt.title('Transmission Spectrum: V1298 Tau b')
        plt.legend()

    def compute_and_plot_mmr(self):
        """
        Compute and plot molecular mass ratio (MMR) profiles for the atmosphere.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for species, mmr in self.spectral_model.mass_fractions.items():
            if species in self.spectral_model.line_species:
                ax.loglog(mmr, self.spectral_model.pressures * 1e-6, label=species)
            elif species in self.spectral_model.model_parameters['filling_species']:
                ax.loglog(mmr, self.spectral_model.pressures * 1e-6, label=species, ls=':')

        total_mmr = np.sum(list(self.spectral_model.mass_fractions.values()), axis=0)
        ax.loglog(total_mmr, self.spectral_model.pressures * 1e-6, label=r'$\sum$ MMR', color='k')

        ax.set_ylim([1e2, 1e-6])
        ax.set_xlabel('MMR')
        ax.set_ylabel('Pressure [bar]')
        ax.set_title('Molecular Mass Ratio Profiles')
        ax.legend()
        plt.show()

    def plot_retrieval_with_model(self, waveleng_retrieval, transit_radii_retrieval):
        """
        Plot retrieved spectrum data along with the model spectrum.

        Parameters:
            waveleng_retrieval (np.ndarray): Wavelength array from retrieval data.
            transit_radii_retrieval (np.ndarray): Transit radius array from retrieval data.
        """
        plt.plot(self.wavelengths[0] * 1e4,
                 (self.transit_radii[0] / self.V1298.star_radius) ** 2,
                 label="Model")
        plt.plot(waveleng_retrieval, transit_radii_retrieval, label="Retrieval")
        plt.xscale("log")
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('$(R_p / R_s)^2$')
        plt.legend()
        plt.show()

    def data(self):
        """
        Load observational data from the specified directory using external function `plot_all_files_in_directory`.
        """
        self.data_ = f.plot_all_files_in_directory(self.data_path, plot=False)

    def plot_model_with_error(self):
        """
        Prepare data with errors for plotting (calls external function `create_real_data`).
        """
        self.JWST_wave, self.JWST_rat, _, _ = f.create_real_data(
            self.R, self.filtered_wavelengths, self.filtered_transit_radii
        )

    def write_spectrum(self):
        """
        Write the model spectrum data to a text file using an external function `write_spectrum_txt`.
        """
        self.JWST_wave, self.JWST_rat, _, _ = f.create_real_data(
            self.R, self.filtered_wavelengths, self.filtered_transit_radii
        )
        filename = f"Spectrum_V1298TAUB_T={self.temp}_M={round(self.mass / 5.97e27, 2)}_CLOUDS={self.wolk}.txt"
        f.write_spectrum_txt(self.JWST_wave * 1e-4, self.JWST_rat ** 2, filename=filename)


if __name__ == "__main__":
    M_e = 5.97e27  
    
    model = V1298SpectrumModel(
            mass=25 * M_e,
            temp=800,
            wolk=None,
            CO2=1e-4,
            H2O=1e-3,
            SO2=1e-7,
            CH4=1e-6,
            CO=1e-5,
            R=1000,
            data_path="/mnt/d/jasper/input_data/input_data/V1298_Tau_b_data",
            offset=0.0
        )
    model.build_model()
    model.plot_transmission()
    plt.show()




    


