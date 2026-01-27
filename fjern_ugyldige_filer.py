import os
import json

# Funktion til at fjerne filer baseret på fejl
def remove_invalid_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_filepath = os.path.join(root, file)
                mrspectra_filepath = json_filepath.replace('.json', '.mrspectra')

                # Tjek for fejl ved indlæsning af JSON og .mrspectra
                try:
                    with open(json_filepath, 'r') as json_file:
                        json.load(json_file)
                except json.JSONDecodeError:
                    print(f"Fjerner ugyldig JSON-fil: {json_filepath}")
                    os.remove(json_filepath)
                    continue  # Gå til næste fil

                # Tjek for fejl ved indlæsning af .mrspectra
                try:
                    with open(mrspectra_filepath, 'rb') as mrspectra_file:
                        binary_data = mrspectra_file.read()
                        # Dummy check for structure (just to simulate processing)
                        if len(binary_data) < 100:  # For eksempel
                            print(f"Fjerner ugyldig .mrspectra-fil: {mrspectra_filepath}")
                            os.remove(mrspectra_filepath)
                except Exception as e:
                    print(f"Fjerner .mrspectra-fil på grund af fejl: {mrspectra_filepath}, fejl: {e}")
                    if os.path.exists(mrspectra_filepath):
                        os.remove(mrspectra_filepath)

# Definér base directory
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

print("Fjernelse af ugyldige filer er gennemført.")
