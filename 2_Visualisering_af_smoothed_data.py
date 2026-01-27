import os
import pandas as pd
import matplotlib.pyplot as plt

# Definer stien til hvor dine CSV-filer er gemt
csv_directory = "C:\\Users\\tom.zbc\\PycharmProjects\\fungis_bacterias_project"

# Funktion til at finde alle smoothed CSV-filer i directory
def find_smoothed_csv_files(directory):
    return [f for f in os.listdir(directory) if f.startswith('smoothed_output_') and f.endswith('.csv')]

# Finder alle smoothed CSV-filer i det definerede bibliotek
csv_filenames = find_smoothed_csv_files(csv_directory)

# Funktion til at visualisere både original og smoothed data fra en CSV-fil
def visualize_smoothed_data(csv_filename):
    filepath = os.path.join(csv_directory, csv_filename)
    try:
        # Læs CSV-fil
        data = pd.read_csv(filepath)

        # Print nogle af værdierne til debugging
        print(f"Viser første rækker af data fra {csv_filename}:")
        print(data.head())

        # Plot både Original Intensity og Smoothed Intensity
        plt.figure(figsize=(10, 6))

        # Plotter den originale intensitet med blå farve og stor tykkelse
        plt.plot(data['Wavenumber (cm-1)'], data['Original Intensity'], label='Original Intensity', color='blue', alpha=0.7, linestyle='-', linewidth=2)

        # Plotter den smoothede intensitet med rød farve og stiplet linje
        plt.plot(data['Wavenumber (cm-1)'], data['Smoothed Intensity'], label='Smoothed Intensity', color='red', alpha=0.7, linestyle='--', linewidth=1.5)

        # Tilføjer labels og titel
        plt.title(f"Original vs Smoothed Data - {csv_filename}")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.legend(loc='upper right')
        plt.grid(True)

        # Viser plot
        plt.show()

    except Exception as e:
        print(f"Error reading or plotting {csv_filename}: {e}")

# Visualiser hver smoothed CSV-fil fundet i directory
for csv_file in csv_filenames:
    visualize_smoothed_data(csv_file)

print("Visualisering af alle smoothed filer er færdig.")
