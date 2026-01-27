import json
import numpy as np
import matplotlib.pyplot as plt

# Indlæs JSON-filen
filename = "Morganella_Morganii_middle_2_785nm.json"
with open(filename, 'r') as file:
    data = json.load(file)

# Udtrækning af spektroskopidata fra JSON
axes_coords = data['spectraDimensions']['AxesCoords']
shape = data['spectraDimensions']['Shape']
labels = data['spectraDimensions']['Labels']

# Konverter til NumPy arrays for lettere behandling
x_coords = np.array(axes_coords[0])
y_coords = np.array(axes_coords[1])
wavenumbers = np.array(axes_coords[2])

# Skab et 3D-array (eller kun 2D for simplificeret visualisering)
# Eksempel: Viser spektret for den første X-Y position (0,0)
spectra_data = np.random.rand(shape[2])  # Brug dette til at simulere data, indtil de rigtige data er kendt

# Visualisering af spektret for en enkelt X-Y position
plt.figure(figsize=(10, 6))
plt.plot(wavenumbers, spectra_data)
plt.title(f"Spectrum for position (X: {x_coords[0]}, Y: {y_coords[0]})")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()
