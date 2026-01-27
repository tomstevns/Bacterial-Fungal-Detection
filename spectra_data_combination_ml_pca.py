import json
import numpy as np
import struct
import matplotlib.pyplot as plt

# Indlæsning af JSON-fil
json_filename = "Morganella_Morganii_middle_2_785nm.json"
with open(json_filename, 'r') as file:
    json_data = json.load(file)

# Udtrækning af metadata fra JSON-filen
x_coords = np.array(json_data['spectraDimensions']['AxesCoords'][0])
y_coords = np.array(json_data['spectraDimensions']['AxesCoords'][1])
wavenumbers = np.array(json_data['spectraDimensions']['AxesCoords'][2])
shape = json_data['spectraDimensions']['Shape']  # (8, 8, 2401)

# Funktion til at læse .mrspectra-fil (antager binært format)
def read_mrspectra_file(filename):
    try:
        with open(filename, 'rb') as f:
            # Læs binært indhold (vi antager 32-bit floats)
            binary_data = f.read()
            # Konverter de binære data til float (32-bit)
            num_floats = len(binary_data) // 4
            spectra = struct.unpack(f'{num_floats}f', binary_data)
            spectra_array = np.array(spectra)
            # Omform til de dimensioner, der er nævnt i JSON-filen (fx 8x8x2401)
            spectra_array = spectra_array.reshape(shape)
            return spectra_array
    except Exception as e:
        print(f"Error reading .mrspectra file: {e}")
        return None

# Indlæs spektraldata fra .mrspectra-fil
mrspectra_filename = "Morganella_Morganii_middle_2_785nm.mrspectra"
spectra_data = read_mrspectra_file(mrspectra_filename)

if spectra_data is not None:
    # Visualiser et spektrum for en tilfældig X-Y position (0,0)
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, spectra_data[0, 0, :])
    plt.title(f"Spectrum for position (X: {x_coords[0]}, Y: {y_coords[0]})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.show()

    # Herefter kan du kombinere disse data til maskinlæring
    # Du kan flade spektraldataene ud og kombinere med labels
    X = spectra_data.reshape(-1, shape[2])  # Flad arrayet ud i [positioner, spektraldata]
    print("Combined data ready for machine learning!")
else:
    print("Failed to read spectra data.")
