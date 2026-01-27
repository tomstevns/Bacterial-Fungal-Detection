import os
import json
import numpy as np
import struct
import csv


# Funktion til at omdøbe filer ved at erstatte mellemrum og kommaer med understregninger
def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Opret nyt navn ved at erstatte mellemrum og kommaer
            new_filename = filename.replace(" ", "_").replace(",", "_")
            old_filepath = os.path.join(root, filename)
            new_filepath = os.path.join(root, new_filename)

            if old_filepath != new_filepath:
                print(f"Omdøber {old_filepath} til {new_filepath}")
                os.rename(old_filepath, new_filepath)


# Funktion til at læse JSON-fil
def load_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Fejl ved læsning af JSON-fil {filename}: {e}")
        return None


# Funktion til at læse .mrspectra-fil (antager binært format)
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
        print(f"Fejl ved læsning af .mrspectra-fil {filename}: {e}")
        return None


# Funktion til at behandle og gemme spektraldata fra JSON- og .mrspectra-filer
def process_spectral_data(json_filepath, mrspectra_filepath):
    try:
        json_data = load_json_file(json_filepath)
        if not json_data:
            print(f"Springer over {json_filepath} på grund af ugyldige JSON-data.")
            return

        # Check if 'wavenumbers' exist in the expected place in JSON
        if 'spectraDimensions' not in json_data or 'AxesCoords' not in json_data['spectraDimensions'] or len(
                json_data['spectraDimensions']['AxesCoords']) < 3:
            print(f"Springer over {json_filepath}: Mangler eller forkert 'wavenumbers'-struktur.")
            return

        shape = json_data['spectraDimensions']['Shape']
        spectra_data = read_mrspectra_file(mrspectra_filepath, shape)

        if spectra_data is not None:
            wavenumbers = np.array(json_data['spectraDimensions']['AxesCoords'][2], dtype=np.float64)
            valid_range = (wavenumbers >= 200) & (wavenumbers <= 2300)

            # Check if valid wavenumbers exist in the range 200-2300 cm⁻¹
            if not np.any(valid_range):
                print(f"Springer over {json_filepath}: Ingen gyldige bølgetal mellem 200 og 2300 cm⁻¹.")
                return

            spectra_data_filtered = spectra_data[0, 0, :][valid_range].astype(np.float64)

            # Clean the spectrum data, replacing NaNs and infinities with 0
            spectrum_cleaned = np.nan_to_num(spectra_data_filtered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

            output_filename = f"output_spectra_{os.path.basename(json_filepath).replace('.json', '')}.csv"
            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Wavenumber (cm-1)', 'Cleaned Intensity'])
                for i in range(len(wavenumbers[valid_range])):
                    csvwriter.writerow([wavenumbers[valid_range][i], spectrum_cleaned[i]])
            print(f"Data gemt til '{output_filename}'.")
        else:
            print(f"Springer over {mrspectra_filepath} på grund af fejl ved indlæsning.")
    except Exception as e:
        print(f"Fejl ved behandling af {json_filepath} og {mrspectra_filepath}: {e}")


# Biblioteker med filer til behandling
base_dirs = [
    "Bacteria-02-07-24-20240821T153301Z-001",
    "Bacteria 20-06-24-20240821T153254Z-001",
    "Bacteria 26-06-24-20240821T153237Z-001",
    "Bacteria-20240821T153307Z-001"
]

# Omdøb filer i hvert bibliotek for at fjerne mellemrum og kommaer
for base_dir in base_dirs:
    print(f"Renaming files in directory: {base_dir}")
    rename_files_in_directory(base_dir)

# Behandling af filer i hvert bibliotek
for base_dir in base_dirs:
    print(f"Behandler filer i biblioteket: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        json_files = [f for f in files if f.endswith(".json")]
        mrspectra_files = [f for f in files if f.endswith(".mrspectra")]

        for json_file, mrspectra_file in zip(json_files, mrspectra_files):
            json_filepath = os.path.join(root, json_file)
            mrspectra_filepath = os.path.join(root, mrspectra_file)
            print(f"Behandler {json_filepath} og {mrspectra_filepath}")
            process_spectral_data(json_filepath, mrspectra_filepath)

print("Dataindlæsning og forberedelse er gennemført.")
