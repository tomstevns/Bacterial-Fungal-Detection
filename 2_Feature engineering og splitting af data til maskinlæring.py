import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Definér base directory og labels mapping
base_dir = "C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project"
labels_mapping = {
    "Aero_Civiae": 0,
    "Aero_Sobria": 1,
    "Assymptomatic_E._coli": 2,
    "Bacillus_Cereus": 3,
    "Bor_bronchiseptica": 4,
    "C._albicans": 5,
    "e.coli_UTI89": 6,
    "E._coli": 7,
    "E._faecalis": 8,
    "E_coli_-_CFT073": 9,
    "k._pneumonia": 10,
    "Morganella_Morganii": 11,
    "P._aerigimosa": 12,
    "S._aureus": 13,
    "s._saprofyticus": 14,
    "stapholococcus_coagulans": 15
}

# Initialiser lister til X_data og y_labels
X_data = []
y_labels = []

# Forventet længde af glattet data
expected_length = 2101

# Gå gennem alle filer i base directory
for filename in os.listdir(base_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)

        # Antag, at 'Cleaned Intensity' er den relevante kolonne
        cleaned_intensity = df['Cleaned Intensity'].values

        # Interpoler til forventet længde
        if len(cleaned_intensity) < expected_length:
            x_old = np.linspace(0, len(cleaned_intensity) - 1, len(cleaned_intensity))
            f = interp1d(x_old, cleaned_intensity, kind='linear', fill_value='extrapolate')
            x_new = np.linspace(0, len(cleaned_intensity) - 1, expected_length)
            normalized_smoothed = f(x_new)
        else:
            normalized_smoothed = cleaned_intensity[:expected_length]

        if len(normalized_smoothed) == expected_length:
            X_data.append(normalized_smoothed)

            # Bestem label fra filnavn
            label = -1
            for key, value in labels_mapping.items():
                if key in filename:
                    label = value
                    break

            if label != -1:
                y_labels.append(label)
            else:
                print(f"Ugyldig label fundet for {filename}, springer over.")
        else:
            print(f"Springer over {filename}, da længden er {len(normalized_smoothed)}.")

# Konverter X_data og y_labels til numpy-arrays
X_data = np.array(X_data, dtype=np.float32)
y_labels = np.array(y_labels)

# Debug-udskrift af længder
print(f"Længde af X_data: {len(X_data)}, længde af y_labels: {len(y_labels)}")
print(f"Labels i y_labels: {y_labels}")

# Gem X_data og y_labels som numpy-filer
np.save('X_data.npy', X_data, allow_pickle=True)
np.save('y_labels.npy', y_labels)

print(f"X_data shape: {X_data.shape}")
print(f"y_labels shape: {y_labels.shape}")
