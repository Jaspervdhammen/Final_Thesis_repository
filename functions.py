# In this file, general functions used over different files are clustered together
# Created by Jasper van der Hammen
# Date 5-7-2025
#-------------------------------------------------------------------------------------------------------------------------------


import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import os

def create_real_data(
    R: float,
    modelled_wavelength: np.ndarray,
    modelled_flux: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic observational data by adding random noise to modelled wavelength 
    values, simulating observational uncertainties in wavelength and flux.

    Parameters
    ----------
    R : float
        Spectral resolution (R = λ / Δλ).
    modelled_wavelength : np.ndarray
        Array of modelled wavelength values.
    modelled_flux : np.ndarray
        Array of modelled transit radii or flux values corresponding to wavelengths.

    Returns
    -------
    real_wavelength : np.ndarray
        Wavelengths with added random perturbation simulating observational noise.
    real_transit_radii : np.ndarray
        Transit radii without perturbation (identical to modelled_flux here).
    error_lambda : np.ndarray
        Wavelength uncertainties calculated as Δλ = λ / R.
    error_transit : np.ndarray
        Transit radius uncertainties (empty here, no error added).
    """
    delta_lambda = lambda wavelength: wavelength / R

    real_wavelength = []
    real_transit_radii = []
    error_lambda = []
    error_transit = []

    for i in modelled_wavelength:
        real_wavelength.append(random.uniform(-1, 1) * delta_lambda(i) + i)
        error_lambda.append(delta_lambda(i))

    return (
        np.array(real_wavelength),
        np.array(modelled_flux),
        np.array(error_lambda),
        np.array(error_transit),
    )


def plot_function(
    spectral_model, 
    transit_radii=False, 
    Temperature_pressure_profile=False, 
    Mass_distribution=False, 
    Mean_molecular_mass_Pressure_profile=False
):
    """
    Plot various characteristics of a spectral model including transmission spectra, 
    temperature-pressure profiles, mass mixing ratio distributions, and mean molecular mass profiles.

    Parameters
    ----------
    spectral_model : object
        The spectral model instance with attributes and methods needed for plotting.
    transit_radii : bool, optional
        If True, plots the transmission spectrum.
    Temperature_pressure_profile : bool, optional
        If True, plots the temperature vs pressure profile.
    Mass_distribution : bool, optional
        If True, plots the mass mixing ratio profiles of different species.
    Mean_molecular_mass_Pressure_profile : bool, optional
        If True, plots the mean molecular mass as a function of pressure.

    Returns
    -------
    None
    """
    if transit_radii:
        wavelengths, transit_radii = spectral_model.calculate_spectrum(mode='transmission')
        plt.plot(wavelengths[0], transit_radii[0], label="Transmission spectra", color="purple", ls=":")
        plt.legend()
        plt.ylabel("$R_p$ [cm]")
        plt.xlabel("Wavelength [micron]")
        plt.xscale('log')
    
    if Temperature_pressure_profile:
        plt.plot(spectral_model.temperatures, spectral_model.pressures * 1e-6)
        plt.yscale('log')
        plt.ylim([1e2, 1e-6])
        plt.title('Temperature profile')
        plt.xlabel('T [K]')
        plt.ylabel('P [bar]')
    
    if Mass_distribution:
        fig, ax = plt.subplots(figsize=(10, 6))

        for species, mass_fraction in spectral_model.mass_fractions.items():
            if species in spectral_model.line_species:
                ax.loglog(mass_fraction, spectral_model.pressures * 1e-6, label=species)

        for species, mass_fraction in spectral_model.mass_fractions.items():
            if species in spectral_model.model_parameters['filling_species']:
                ax.loglog(mass_fraction, spectral_model.pressures * 1e-6, label=species, ls=':')

        ax.loglog(np.sum(list(spectral_model.mass_fractions.values()), axis=0), spectral_model.pressures * 1e-6, label=r'$\sum$ MMR', color='k')

        ax.set_ylim([1e2, 1e-6])
        ax.set_title('MMR profiles')
        ax.set_xlabel('MMR')
        ax.set_ylabel('P [bar]')
        ax.legend()
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(spectral_model.mean_molar_masses, spectral_model.pressures * 1e-6, label='Mean Molar Mass')

        ax.set_ylim([1e2, 1e-6])
        ax.set_xlabel('Mean molar masses')
        ax.set_ylabel('P [bar]')


def get_all_data(file_path: str) -> pd.DataFrame | None:
    """
    Reads a CSV file containing observational data and selects specific columns.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing selected columns if successful, otherwise None.
    """
    desired_columns = [
        'CENTRALWAVELNG', 'BANDWIDTH', 'PL_TRANDEP', 'PL_TRANDEPERR1', 'PL_TRANDEPERR2',
        'PL_TRANDEP_AUTHORS', 'PL_RATROR', 'PL_RATRORERR1', 'PL_RATRORERR2', 'PL_RATROR_AUTHORS',
        'PL_RADJ', 'PL_RADJERR1', 'PL_RADJERR2', 'PL_RADJ_AUTHORS', 'PL_TRANMID', 'PL_TRANMIDERR1',
        'PL_TRANMIDERR2', 'ST_RAD', 'ST_RADERR1', 'ST_RADERR2', 'ST_RAD_AUTHORS'
    ]
    
    try:
        data = pd.read_csv(file_path, delimiter=',', skipinitialspace=True)
        data_selected = data[desired_columns]
        return data_selected
    except pd.errors.ParserError as e:
        print(f"Error reading CSV file: {e}")
        return None
    except KeyError as e:
        print(f"Column not found in the CSV file: {e}")
        return None


def plot_all_files_in_directory(directory_path: str, plot: bool = True) -> dict:
    """
    Reads all CSV files in a directory, extracts observational data, and optionally plots the spectra with error bars.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing CSV files.
    plot : bool, optional
        Whether to plot the data from each file.

    Returns
    -------
    dict
        Dictionary with filenames as keys and their corresponding data as values.
    """
    all_files_data = {}

    files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if plot:
        plt.figure(figsize=(10, 6))

    for file in files:
        file_path = os.path.join(directory_path, file)
        all_data = get_all_data(file_path)

        if all_data is not None:
            all_files_data[file] = all_data
            if plot:
                x = all_data['CENTRALWAVELNG']
                y = all_data['PL_RATROR']**2
                xerr = all_data["BANDWIDTH"]
                yerr_lower = abs(all_data['PL_RATRORERR2']*2)
                yerr_upper = all_data['PL_RATRORERR1']*2
                
                plt.plot(x, y, 'o', color='tab:purple', alpha=0.5)
                plt.fill_between(
                    x,
                    y - yerr_lower,
                    y + yerr_upper,
                    color='tab:purple',
                    alpha=0.3
                )

    if plot:
        plt.xlabel('Central Wavelength')
        plt.ylabel('$(R_{planet}/R_{star} )^2$]')
        plt.title('Transmission spectra of different sources')
        plt.legend()

    return all_files_data


def write_spectrum_txt(wavelengths, fluxes, filename="spectrum.txt"):
    """
    Write wavelength and flux data to a text file in a format compatible with petitRADTRANS.
    Wavelengths are converted to Angstroms and output is sorted and in decimal notation.

    Parameters
    ----------
    wavelengths : list or np.ndarray
        List or array of wavelengths in microns.
    fluxes : list or np.ndarray
        Corresponding flux values.
    filename : str, optional
        Output filename (default is "spectrum.txt").

    Raises
    ------
    ValueError
        If the input wavelength and flux arrays have different lengths.
    """
    if not (len(wavelengths) == len(fluxes)):
        raise ValueError("Wavelengths and fluxes must have the same length.")

    sorted_data = sorted(zip(np.array(wavelengths)*1e4, fluxes), key=lambda x: x[0])

    with open(filename, 'w') as f:
        f.write("# Wavelength [micron], Flux [(Rp/Rstar)^2] \n")
        for wl, fl in sorted_data:
            f.write(f"{wl:.6f} {fl:.10f}\n")

    print(f"File '{filename}' written successfully.")


def Instrument_discrimination(
    wavelengths: np.ndarray,
    y_coords: np.ndarray,
    MIRI=True,
    NIRSPEC=True,
    NIRCAM=True,
    all=False
) -> Tuple[list, list]:
    """
    Filter input wavelength and y-coordinate data based on the wavelength coverage of JWST instruments.

    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelength values.
    y_coords : np.ndarray
        Corresponding y-axis values.
    MIRI : bool, optional
        Include MIRI wavelength range (default True).
    NIRSPEC : bool, optional
        Include NIRSPEC wavelength range (default True).
    NIRCAM : bool, optional
        Include NIRCAM wavelength range (default True).
    all : bool, optional
        Return all data without filtering (default False).

    Returns
    -------
    filtered_wavelengths : list
        Filtered wavelength values based on instrument coverage.
    filtered_y : list
        Corresponding filtered y-axis values.
    """
    instrument_bounds = []
    if MIRI:
        instrument_bounds.append([4.61, 13.628])
    if NIRSPEC:
        instrument_bounds.append([2.73551, 4.81311])
    if NIRCAM:
        instrument_bounds.append([2.4572, 5.0375])
    
    if all:
        return list
