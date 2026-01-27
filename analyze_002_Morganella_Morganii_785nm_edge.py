import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import struct
import csv

# Indlæsning af JSON-fil
json_filename = "Morganella_Morganii_785nm_edge.json"
with open(json_filename, 'r') as file:
    json_data = json.load(file)

# Udtrækning af metadata fra JSON-filen
x_coords = np.array(json_data['spectraDimensions']['AxesCoords'][0], dtype=np.float64)
y_coords = np.array(json_data['spectraDimensions']['AxesCoords'][1], dtype=np.float64)
wavenumbers = np.array(json_data['spectraDimensions']['AxesCoords'][2], dtype=np.float64)
shape = json_data['spectraDimensions']['Shape']  # (8, 8, 2401)


# Funktion til at læse .mrspectra-fil (antager binært format)
def read_mrspectra_file(filename):
    try:
        with open(filename, 'rb') as f:
            binary_data = f.read()
            num_floats = len(binary_data) // 4
            spectra = struct.unpack(f'{num_floats}f', binary_data)
            spectra_array = np.array(spectra, dtype=np.float64)
            spectra_array = spectra_array.reshape(shape)
            return spectra_array
    except Exception as e:
        print(f"Error reading .mrspectra file: {e}")
        return None


# Indlæs spektraldata fra .mrspectra-fil
mrspectra_filename = "Morganella_Morganii_785nm_edge.mrspectra"
spectra_data = read_mrspectra_file(mrspectra_filename)

if spectra_data is not None:
    # Begræns data til intervallet 200-2300 cm⁻¹
    valid_range = (wavenumbers >= 200) & (wavenumbers <= 2300)
    wavenumbers_filtered = wavenumbers[valid_range].astype(np.float64)
    spectrum_filtered = spectra_data[0, 0, :][valid_range].astype(np.float64)

    # Rens spektraldata for NaN og inf
    spectrum_cleaned = np.nan_to_num(spectrum_filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    # Justerede vertikale intensiteter for specifikke wavenumbers med glidende overgang
    intensity_adjustments = {
        200: 0.57,
        2300: 0.2
    }

    # Definer en multimodal Lorentzian funktion (sum af flere Lorentz-kurver) med baseline
    def multimodal_lorentzian(params, x):
        baseline = params[0]  # Første parameter er baseline
        y = np.full_like(x, baseline, dtype=np.float64)

        n_peaks = (len(params) - 1) // 3  # Hver peak har 3 parametre (amp, cen, wid)
        for i in range(n_peaks):
            amp = params[i * 3 + 1]
            cen = params[i * 3 + 2]
            wid = params[i * 3 + 3]
            y += amp * (wid ** 2 / ((x - cen) ** 2 + wid ** 2))

        # Anvend intensitetsjusteringer med glidende overgang
        for wn, adjustment in intensity_adjustments.items():
            mask = np.exp(-((x - wn) / 10) ** 2)  # Glidende overgang over 10 cm⁻¹
            y *= (1 - mask + mask * adjustment)

        return y

    # Fejlberegning for kurvefitting
    def objective_function(params):
        fitted_curve = multimodal_lorentzian(params, wavenumbers_filtered)
        return np.sum((fitted_curve - spectrum_cleaned) ** 2)

    # Vi indstiller grænserne til de opdaterede peaks, med baseline som første parameter
    important_peaks = [400, 550, 700, 750, 800, 1000, 1300, 1470, 1700]  # Peaks fra tidligere
    bounds = [(0, 100)]  # Baseline kan være fra 0 til 100

    # Tilføj grænser for hver peak (amplitude, center, width)
    for peak in important_peaks:
        bounds += [
            (0, 1.5 * max(spectrum_cleaned)),  # Amplitude
            (peak - 5, peak + 5),  # Center tættere på peak (smallere interval)
            (5, 50)  # Justeret bredde for bedre fleksibilitet
        ]

    print(f"Bounds for parameters: {bounds}")

    try:
        # Nelder-Mead optimering med baseline og opdaterede peaks
        initial_params = [1.0] + [val for peak in important_peaks for val in (0.5, peak, 20)]  # Startværdier
        result = minimize(objective_function, initial_params, method='Nelder-Mead')

        # Udtræk de optimale parametre fra resultatet
        optimal_params = result.x
        print(f"Optimal parameters found: {optimal_params}")

        # Generer den fitte kurve
        fitted_curve = multimodal_lorentzian(optimal_params, wavenumbers_filtered)

        # Beregn residualerne (forskellen mellem oprindelig og fittet kurve)
        residuals = spectrum_cleaned - fitted_curve

        # Gem resultatet i en CSV-fil
        output_filename = "spectra_fitted_output_Morganella_Morganii_785nm_edge.csv"
        with open(output_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Wavenumber (cm-1)', 'Original Intensity', 'Fitted Intensity', 'Residual'])
            for i in range(len(wavenumbers_filtered)):
                csvwriter.writerow([wavenumbers_filtered[i], spectrum_cleaned[i], fitted_curve[i], residuals[i]])

        print(f"Output file '{output_filename}' generated.")

        # Visualisering af det originale spektrum og den multimodale fitte kurve
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumbers_filtered, spectrum_cleaned, 'b-', label='Original data')
        plt.plot(wavenumbers_filtered, fitted_curve, 'r--', label='Fitted curve (multimodal)')

        plt.title(f"Multimodal Curve Fitting with Smoothed Adjusted Intensities")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid(True)  # Kun grid, ingen vertikale stiplede linjer
        plt.show()

    except Exception as e:
        print(f"Error during fitting: {e}")

else:
    print("Failed to read spectra data.")
