import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import struct

# Funktion til at læse JSON-fil
def read_json_file(json_filename):
    try:
        with open(json_filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error reading JSON file: {json_filename} not found.")
        return None

# Funktion til at læse .mrspectra-fil
def read_mrspectra_file(filename, shape):
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

# Funktion til at læse fittede outputfiler fra CSV
def read_fitted_spectra(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error reading fitted spectra file: {filename} not found.")
        return None

# Dummy funktion til at træne ML-modeller (her kan du bruge din faktiske model)
def train_model(X_train, y_train, X_test):
    # Dummy forudsigelser - erstat med din ML-model
    return np.random.choice(np.unique(y_train), size=len(X_test))

# Filoplysninger for de to bakterier og de fittede spektrale outputfiler
bacteria_files = [
    {
        "mrspectra": "Morganella_Morganii_middle_2_785nm.mrspectra",
        "json": "Morganella_Morganii_middle_2_785nm.json",
        "fitted_spectra": "spectra_fitted_output_middle.csv",  # Denne fil skal genereres fra tidligere analyse
        "shape": (8, 8, 2401)
    },
    {
        "mrspectra": "Morganella_Morganii_785nm_edge.mrspectra",
        "json": "Morganella_Morganii_785nm_edge.json",
        "fitted_spectra": "spectra_fitted_output_edge.csv",  # Denne fil skal også genereres
        "shape": (8, 8, 2401)
    }
]

# Samlede resultater for alle bakterier
y_true_all = []
y_pred_all = []

# Loop gennem hver bakterie for at køre analysen
for files in bacteria_files:
    # Indlæs JSON- og .mrspectra-filer
    json_data = read_json_file(files["json"])
    spectra_data = read_mrspectra_file(files["mrspectra"], files["shape"])
    fitted_data = read_fitted_spectra(files["fitted_spectra"])

    # Check om alle nødvendige data blev læst korrekt
    if spectra_data is None or fitted_data is None:
        print(f"Skipping {files['fitted_spectra']} as necessary data could not be read.")
        continue  # Spring denne fil over, hvis der er problemer

    # Udtræk relevante data fra JSON og fittede spektrale filer (forenklet her, tilpas efter behov)
    X = fitted_data[['Wavenumber', 'Fitted Intensity']].values  # Eksempel på feature-formatering
    y = np.random.randint(0, 2, X.shape[0])  # Dummy labels (erstat med faktiske labels)

    # Split data i trænings- og test-sæt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Træn modellen og få forudsigelser
    y_pred = train_model(X_train, y_train, X_test)

    # Saml resultaterne
    y_true_all.extend(y_test)  # Faktiske labels
    y_pred_all.extend(y_pred)  # Forudsigelser

# Konverter resultaterne til arrays
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# Check om der er nok data til at generere confusion matrix
if len(y_true_all) == 0 or len(y_pred_all) == 0:
    print("No valid data to generate confusion matrix.")
else:
    # Generer confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)

    # Visualiser confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Two Morganella Bacteria Models')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
