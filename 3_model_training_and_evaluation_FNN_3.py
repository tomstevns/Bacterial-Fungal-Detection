import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

# Indlæs billeder fra en mappe
def load_images_from_folder(folder, target_size=(100, 100)):
    images = []
    labels = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith('.jpg'):
                img = load_img(os.path.join(root, filename), target_size=target_size)
                img_array = img_to_array(img) / 255.0  # Normaliser til [0, 1]
                images.append(img_array)
                labels.append(get_label_from_filename(filename))
                print(f"Indlæst billede: {filename}, dimension: {img_array.shape}")  # Debugging
    return np.array(images), np.array(labels)

# Implementer en funktion til at konvertere filnavn til label
def get_label_from_filename(filename):
    return filename.split('_')[1].split('.')[0]  # Juster logik efter dit filnavnekonvention

# Biblioteker med filer til behandling
base_dirs = [
    "Bacteria-02-07-24-20240821T153301Z-001",
    "Bacteria 20-06-24-20240821T153254Z-001",
    "Bacteria 26-06-24-20240821T153237Z-001",
    "Bacteria-20240821T153307Z-001"
]

# Indlæs X_data og y_labels fra alle base_dirs
all_images = []
all_labels = []
for base_dir in base_dirs:
    print(f"Indlæser billeder fra: {base_dir}")
    X_data, y_labels = load_images_from_folder(base_dir)

    # Tjek for tomme arrays
    if X_data.size > 0:
        all_images.append(X_data)
        all_labels.append(y_labels)

# Sammenkæd alle billeder og labels, hvis de er tilgængelige
if all_images:
    X_data = np.concatenate(all_images)
    y_labels = np.concatenate(all_labels)

    print(f"X_data shape: {X_data.shape}")
    print(f"y_labels shape: {y_labels.shape}")

    # Print unikke klasser
    unique_classes = np.unique(y_labels)
    print(f"Unikke klasser: {unique_classes}")
    print(f"Antal klasser: {len(unique_classes)}")

    # Definér alle klasser
    num_classes = len(unique_classes)
    class_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
    y_labels_mapped = np.vectorize(class_mapping.get)(y_labels)

    # Konverter labels til kategorisk format
    y_labels_categorical = to_categorical(y_labels_mapped, num_classes=num_classes)

    # Beregn class_weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.arange(num_classes), y=y_labels_mapped)
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    # KFold krydsvalidering
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_labels_mapped[train_index], y_labels_mapped[test_index]

        # Konverter labels til kategorisk format
        y_train_categorical = to_categorical(y_train, num_classes=num_classes)
        y_test_categorical = to_categorical(y_test, num_classes=num_classes)

        # Byg model med konvolutionslag
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Kompiler modellen
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Early stopping og læringsrate-reduktion
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

        # Træn modellen uden data augmentering
        history = model.fit(X_train, y_train_categorical, batch_size=16,
                            epochs=50, validation_data=(X_test, y_test_categorical),
                            callbacks=[early_stopping, reduce_lr], class_weight=class_weights_dict)

        # Evaluér modellen på testdatasættet
        loss, accuracy = model.evaluate(X_test, y_test_categorical)
        fold_accuracies.append(accuracy)
        print(f"Fold Accuracy: {accuracy}")

        # Forudsig klasser på testdatasættet
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test_categorical, axis=1)

        # Beregn og print confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred_classes)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Vis confusion matrix grafisk
        ConfusionMatrixDisplay(conf_matrix).plot(cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

    # Gennemsnitlig nøjagtighed over alle folds
    print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
else:
    print("Ingen billeder blev indlæst fra de angivne mapper.")
