import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from keras.optimizers import Adam

# Antag at X_data og y_labels allerede er korrekt genereret i 2_..
X_data = np.load('X_data.npy')  # Brug din faktiske outputfil her
y_labels = np.load('y_labels.npy')

# Normalisering af data hvis nødvendigt
X_data = X_data / np.max(X_data)

# Step 1: Split datasæt i trænings- og testdatasæt
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

# Step 2: Konverter labels til kategoriske formater
num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Step 3: Byg en CNN-model til klassifikation
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Step 4: Kompiler modellen med en mindre learning rate
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Træn modellen
model.fit(X_train, y_train_categorical, validation_data=(X_test, y_test_categorical), epochs=50, batch_size=32)

# Step 6: Evaluér modellen på testdatasættet
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Model Accuracy: {accuracy}")

# Gem modellen hvis ønsket
model.save("cnn_model.h5")
