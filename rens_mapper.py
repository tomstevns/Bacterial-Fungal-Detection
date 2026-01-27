import os
import shutil

# Definer de forventede klasser
expected_classes = ['Aero', 'Bacillus', 'Bor', 'E', 'Morganella', 'stapholococcus']

# Funktion til at fjerne ikke-forventede klasser fra mapperne
def clean_data_folders(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name not in expected_classes:
                dir_path = os.path.join(root, dir_name)
                print(f"Fjerner ikke-forventet klassemappe: {dir_path}")
                shutil.rmtree(dir_path)  # Fjern hele mappen

# Rigtige pathes til dine mapper
train_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/train'
validation_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/validation'
test_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/test'

# Rens mapperne
clean_data_folders(train_dir)
clean_data_folders(validation_dir)
clean_data_folders(test_dir)

print("Rensning af mapperne er fuldført.")
