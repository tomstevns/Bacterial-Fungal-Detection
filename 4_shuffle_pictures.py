import os
import shutil
import random

# Funktion til at tømme mapper
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Fjern kun filer, ikke undermapper

# Kopier JPG-filer til de rette mapper
def organize_images(source_dir, train_dir, validation_dir, test_dir, train_ratio=0.7, validation_ratio=0.15):
    # Opret mapperne, hvis de ikke eksisterer
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Tøm mapperne kun for filer, der findes i dem
    clear_directory(train_dir)
    clear_directory(validation_dir)
    clear_directory(test_dir)

    # Find alle JPG-filer i kilde-mappen
    files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    if len(files) == 0:
        print("Ingen billeder fundet i kildemappen.")
        return

    # Bland filerne tilfældigt
    random.shuffle(files)

    # Beregn antal filer til træning, validering og test
    train_count = int(len(files) * train_ratio)
    validation_count = int(len(files) * validation_ratio)

    # Kopier filer til de rette mapper
    for i, filename in enumerate(files):
        src = os.path.join(source_dir, filename)

        if i < train_count:
            dst = os.path.join(train_dir, filename)
        elif i < train_count + validation_count:
            dst = os.path.join(validation_dir, filename)
        else:
            dst = os.path.join(test_dir, filename)

        print(f"Kopierer {src} til {dst}")
        shutil.copy(src, dst)  # Brug copy() i stedet for move()

# Definer stierne
source_dir = 'data'  # Mappen, hvor dine JPG-filer ligger
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# Kald funktionen for at organisere billederne
organize_images(source_dir, train_dir, validation_dir, test_dir)
