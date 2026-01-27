import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Indlæsning af JSON-fil
json_filename = "Morganella_Morganii_middle_2_785nm.json"
with open(json_filename, 'r') as file:
    json_data = json.load(file)

# Udtrækning af spektrale dimensioner og metadata fra JSON
x_coords = np.array(json_data['spectraDimensions']['AxesCoords'][0])
y_coords = np.array(json_data['spectraDimensions']['AxesCoords'][1])
wavenumbers = np.array(json_data['spectraDimensions']['AxesCoords'][2])
shape = json_data['spectraDimensions']['Shape']

# Simuleret indlæsning af .mrspectra data - antag at hver position (X, Y) har et spektrum
# (Dette skal udskiftes med din faktiske metode til at hente spektraldata fra en .mrspectra fil)
def load_mrspectra_data():
    # Simuler tilfældige spektrale data for alle (X, Y) positioner
    return np.random.rand(shape[0], shape[1], shape[2])

spectra_data = load_mrspectra_data()

# Eksempel: Visualisering af spektrummet for den første X-Y position (0,0)
plt.figure(figsize=(10, 6))
plt.plot(wavenumbers, spectra_data[0, 0, :])
plt.title(f"Spectrum for position (X: {x_coords[0]}, Y: {y_coords[0]})")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()

# Forberedelse til maskinlæringsmodellen
# Vi flader X, Y positionsdata og spektraldata ud i en feature-matrix
X = spectra_data.reshape(-1, shape[2])  # Flad arrayet ud i [positions, spektraldata]

# Simuleret label (klasserne for de 13 bakterier og vira)
# I en rigtig situation ville du have bakterie-/virusklassifikationer for hvert spektrum
y = np.random.randint(0, 13, size=(X.shape[0],))  # Simuler tilfældige labels for nu

# Split data i træning og test sæt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn en simpel maskinlæringsmodel (Random Forest som eksempel)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Forudsig på testdata
y_pred = clf.predict(X_test)

# Evaluer modellens nøjagtighed
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Forberedelse til fremtidig modeltræning med rigtige labels
