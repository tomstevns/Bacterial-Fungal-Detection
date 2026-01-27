import struct
import matplotlib.pyplot as plt

# Indlæsning af *.mrspectra fil som binær
filename = "Morganella_Morganii_middle_2_785nm.mrspectra"

try:
    # Åbn filen som en binær fil
    with open(filename, 'rb') as file:
        # Læs alt indhold fra filen som binær data
        binary_data = file.read()

    # Antag, at dataene er gemt som float-32 (4 bytes per værdi)
    num_floats = len(binary_data) // 4
    spectrum = struct.unpack(f'{num_floats}f', binary_data)

except Exception as e:
    print(f"Failed to read or process the file: {e}")
    exit()

# Plotting af spektret
plt.figure(figsize=(10, 6))
plt.plot(spectrum)
plt.title("MR Spectra - Visualized (Binary Data)")
plt.xlabel("Data Points")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()
