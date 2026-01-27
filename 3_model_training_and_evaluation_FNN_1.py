import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Indlæs X_data og y_labels fra de gemte .npy filer
try:
    X_data = np.load('X_data.npy')
    y_labels = np.load('y_labels.npy')
    print(f"X_data shape: {X_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")
except FileNotFoundError as e:
    print(f"Fejl: {e}")
    exit()

# Del datasættet op i trænings- og testdata
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Konverter labels til kategorisk format
num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Byg en simpel FNN-model (fully connected neural network)
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Kompiler modellen
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Træn modellen
model.fit(X_train, y_train_categorical, epochs=10, batch_size=32)

# Evaluér modellen på testdatasættet
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Model Accuracy: {accuracy}")
