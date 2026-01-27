import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Definér base directory og labels mapping
base_dir = "C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project"
output_dir = "C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data"
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

# Minimum samples per class requirement
min_samples_per_class = 2
augmentation_factor = 5  # Hvor mange samples der skal genereres ved augmentering

# Initialiser lister til X_data og y_labels
X_data = []
y_labels = []

# Forventet længde af glattet data
expected_length = 2101

# Tæl samples per klasse for validering
samples_count = {key: 0 for key in labels_mapping.keys()}


# Funktion til at lave data augmentation ved at tilføje støj og variation
def augment_data(data, num_augmented, noise_factor=0.01):
    augmented_data = []
    for _ in range(num_augmented):
        noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
        augmented_sample = data + noise
        augmented_data.append(augmented_sample)
    return augmented_data


# Gå gennem alle filer i base directory
for filename in os.listdir(base_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)

        # Antag, at 'Cleaned Intensity' er den relevante kolonne
        if 'Cleaned Intensity' in df.columns:
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
                # Bestem label fra filnavn
                label = -1
                for key, value in labels_mapping.items():
                    if key in filename:
                        label = value
                        samples_count[key] += 1
                        break

                if label != -1:
                    X_data.append(normalized_smoothed)
                    y_labels.append(label)
                else:
                    print(f"Ugyldig label fundet for {filename}, springer over.")
            else:
                print(f"Springer over {filename}, da længden er {len(normalized_smoothed)}.")
        else:
            print(f"Springer over {filename}, da kolonnen 'Cleaned Intensity' mangler.")

# Udfør augmentering for klasser med for få samples
for key, count in samples_count.items():
    if count < min_samples_per_class:
        needed_augments = min_samples_per_class - count
        print(
            f"Advarsel: Klasse '{key}' har kun {count} samples, kræver mindst {min_samples_per_class}. Udfører data augmentering...")

        # Find det sample, der skal augmenteres
        for i in range(len(X_data)):
            if y_labels[i] == labels_mapping[key]:
                augmented_samples = augment_data(X_data[i], needed_augments)
                X_data.extend(augmented_samples)
                y_labels.extend([labels_mapping[key]] * needed_augments)
        print(f"Tilføjet {needed_augments} nye samples til klasse '{key}'.")

# Konverter X_data og y_labels til numpy-arrays
X_data = np.array(X_data, dtype=np.float32)
y_labels = np.array(y_labels)

# Debug-udskrift af længder
print(f"Længde af X_data: {len(X_data)}, længde af y_labels: {len(y_labels)}")
print(f"Labels i y_labels: {y_labels}")

# Opret output directory, hvis det ikke findes
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Gem X_data og y_labels som numpy-filer
np.save(os.path.join(output_dir, 'X_data_augmented.npy'), X_data, allow_pickle=True)
np.save(os.path.join(output_dir, 'y_labels_augmented.npy'), y_labels)

print(f"X_data shape: {X_data.shape}")
print(f"y_labels shape: {y_labels.shape}")

# Verifikation af augmentering
for key, count in samples_count.items():
    total_samples = np.sum(np.array(y_labels) == labels_mapping[key])
    if total_samples >= min_samples_per_class:
        print(f"Klasse '{key}' har nu {total_samples} samples efter augmentering.")
    else:
        print(f"Klasse '{key}' har stadig ikke nok samples ({total_samples}) efter augmentering.")
