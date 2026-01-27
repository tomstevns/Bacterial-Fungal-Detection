import os
import json
import numpy as np
import struct
import csv
from scipy.interpolate import interp1d

# Funktion til at omdøbe filer
def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            new_filename = filename.replace(" ", "_").replace(",", "_")
            old_filepath = os.path.join(root, filename)
            new_filepath = os.path.join(root, new_filename)

            if old_filepath != new_filepath:
                print(f"Omdøber {old_filepath} til {new_filepath}")
                os.rename(old_filepath, new_filepath)

# Funktion til at fjerne ugyldige filer
def remove_invalid_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_filepath = os.path.join(root, file)
                mrspectra_filepath = json_filepath.replace('.json', '.mrspectra')

                try:
                    with open(json_filepath, 'r') as json_file:
                        json.load(json_file)
                except json.JSONDecodeError:
                    print(f"Fjerner ugyldig JSON-fil: {json_filepath}")
                    os.remove(json_filepath)
                    continue  # Gå til næste fil

                try:
                    with open(mrspectra_filepath, 'rb') as mrspectra_file:
                        binary_data = mrspectra_file.read()
                        if len(binary_data) < 100:  # Dummy check for struktur
                            print(f"Fjerner ugyldig .mrspectra-fil: {mrspectra_filepath}")
                            os.remove(mrspectra_filepath)
                except Exception as e:
                    print(f"Fjerner .mrspectra-fil på grund af fejl: {mrspectra_filepath}, fejl: {e}")
                    if os.path.exists(mrspectra_filepath):
                        os.remove(mrspectra_filepath)

# Funktion til at læse JSON-fil
def load_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Fejl ved læsning af JSON-fil {filename}: {e}")
        return None

# Funktion til at læse .mrspectra-fil
def read_mrspectra_file(filename, shape):
    try:
        with open(filename, 'rb') as f:
            binary_data = f.read()
            num_floats = len(binary_data) // 4
            spectra = struct.unpack(f'{num_floats}f', binary_data)
            spectra_array = np.array(spectra, dtype=np.float64)
            return spectra_array.reshape(shape)
    except Exception as e:
        print(f"Fejl ved læsning af .mrspectra-fil {filename}: {e}")
        return None

# Funktion til at rense data
def clean_spectrum(spectrum):
    return np.nan_to_num(spectrum, nan=0.0)

# Interpolationsfunktion til spektrum
def interpolate_to_expected_length(spectrum, expected_length):
    if len(spectrum) < expected_length:
        x_old = np.linspace(0, len(spectrum) - 1, len(spectrum))
        f = interp1d(x_old, spectrum, kind='linear', fill_value='extrapolate')
        x_new = np.linspace(0, len(spectrum) - 1, expected_length)
        return f(x_new)
    else:
        return spectrum[:expected_length]

# Funktion til at interpolere spektrum til forventet form
def interpolate_spectrum_to_shape(spectrum, target_shape):
    target_size = np.prod(target_shape)
    x_old = np.linspace(0, len(spectrum) - 1, len(spectrum))
    x_new = np.linspace(0, len(spectrum) - 1, target_size)
    interpolated_spectrum = np.interp(x_new, x_old, spectrum)
    return interpolated_spectrum.reshape(target_shape)

# Opdateret process_spectral_data funktion med håndtering af uventede størrelser
def process_spectral_data(json_filepath, mrspectra_filepath, expected_shape=(8, 8, 2001), expected_length=2101):
    json_data = load_json_file(json_filepath)
    if not json_data:
        print(f"Springer over {json_filepath} på grund af ugyldige JSON-data.")
        return

    # Kontrollér at JSON-strukturen er som forventet
    if 'spectraDimensions' not in json_data or 'AxesCoords' not in json_data['spectraDimensions']:
        print(f"Springer over {json_filepath}: Mangler eller forkert 'wavenumbers'-struktur.")
        return

    # Hent og læs spektraldata
    shape = json_data['spectraDimensions']['Shape']
    spectra_data = read_mrspectra_file(mrspectra_filepath, shape)

    # Hvis læsning mislykkedes, prøv at interpolere data til den forventede form
    if spectra_data is None:
        print(f"Prøver at interpolere data for {mrspectra_filepath} på grund af uventet størrelse.")
        try:
            with open(mrspectra_filepath, 'rb') as f:
                binary_data = f.read()
                num_floats = len(binary_data) // 4
                spectra = struct.unpack(f'{num_floats}f', binary_data)
                spectra_array = np.array(spectra, dtype=np.float64)
                spectra_data = interpolate_spectrum_to_shape(spectra_array, expected_shape)
        except Exception as e:
            print(f"Kunne ikke interpolere {mrspectra_filepath}: {e}")
            return

    # Filtrér og interpolér til forventet længde
    wavenumbers = np.array(json_data['spectraDimensions']['AxesCoords'][2], dtype=np.float64)
    valid_range = (wavenumbers >= 200) & (wavenumbers <= 2300)
    spectra_data_filtered = spectra_data[0, 0, :][valid_range].astype(np.float64)
    spectrum_cleaned = clean_spectrum(spectra_data_filtered)
    spectrum_interpolated = interpolate_to_expected_length(spectrum_cleaned, expected_length)

    # Justér wavenumbers og interpoleret spektrum til samme længde, hvis de ikke passer
    min_length = min(len(wavenumbers[valid_range]), len(spectrum_interpolated))
    wavenumbers_adjusted = wavenumbers[valid_range][:min_length]
    spectrum_interpolated_adjusted = spectrum_interpolated[:min_length]

    # Gem som CSV
    output_filename = f"output_spectra_{os.path.basename(json_filepath).replace('.json', '')}.csv"
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Wavenumber (cm-1)', 'Cleaned Intensity'])
        for i in range(min_length):
            csvwriter.writerow([wavenumbers_adjusted[i], spectrum_interpolated_adjusted[i]])
    print(f"Data gemt til '{output_filename}'.")

# Biblioteker med filer til behandling
base_dirs = [
    "Bacteria-02-07-24-20240821T153301Z-001",
    "Bacteria 20-06-24-20240821T153254Z-001",
    "Bacteria 26-06-24-20240821T153237Z-001",
    "Bacteria-20240821T153307Z-001"
]

# Gå gennem alle biblioteker for at fjerne ugyldige filer
for base_dir in base_dirs:
    print(f"Fjerner ugyldige filer i biblioteket: {base_dir}")
    remove_invalid_files(base_dir)

# Omdøb filer i hvert bibliotek for at fjerne mellemrum og kommaer
for base_dir in base_dirs:
    print(f"Renaming files in directory: {base_dir}")
    rename_files_in_directory(base_dir)

# Behandling af filer i hvert bibliotek
for base_dir in base_dirs:
    print(f"Behandler filer i biblioteket: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json")]
        for json_file in json_files:
            json_filepath = os.path.join(root, json_file)
            mrspectra_file = json_file.replace('.json', '.mrspectra')
            mrspectra_filepath = os.path.join(root, mrspectra_file)
            print(f"Behandler {json_filepath} og {mrspectra_filepath}")
            process_spectral_data(json_filepath, mrspectra_filepath)

print("Dataindlæsning og forberedelse er gennemført.")
