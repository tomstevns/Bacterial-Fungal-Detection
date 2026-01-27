import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# Funktion til at ekstrahere labels fra filnavnene
def extract_label_from_filename(filename):
    if 'Morganella_Morganii' in filename:
        return 0
    elif 'Aero_Sobria' in filename:
        return 1
    # Tilføj flere regler for de andre bakterier her
    else:
        return -1  # Ukendt label

# Simulér filnavne (opdater denne del, hvis du har rigtige filnavne)
filenames = ['output_spectra_Morganella_Morganii_middle_785nm.csv',
             'output_spectra_Aero_Sobria_middle_785nm.csv']

# Ekstraher labels fra filnavne
y_labels = [extract_label_from_filename(f) for f in filenames]

# Konverter til numpy array
y_labels = np.array(y_labels)

# Fjern evt. ukendte labels (-1)
valid_indices = y_labels != -1
y_labels = y_labels[valid_indices]
X_data = X_data[valid_indices]  # Sørg for at X_data er defineret korrekt

# Step 1: Split datasæt i trænings- og testdatasæt
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Step 2: Konverter labels til kategoriske formater
num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Step 3: Byg en simpel FNN-model
model = Sequential()

# Tilføj lag til din FNN
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Kompiler modellen
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Træn modellen
model.fit(X_train, y_train_categorical, epochs=10, batch_size=32)

# Step 5: Evaluér modellen på testdatasættet
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Model Accuracy: {accuracy}")
