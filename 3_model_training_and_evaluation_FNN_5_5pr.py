import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Utility function to check data integrity
def check_data_integrity(data_dir, expected_classes, min_samples_per_class=1):
    """
    Checks if the data directory contains all the expected classes
    and that each class has the minimum number of samples.

    :param data_dir: Path to the data directory
    :param expected_classes: List of expected class names
    :param min_samples_per_class: Minimum number of samples expected per class
    :return: None
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    class_dirs = os.listdir(data_dir)
    missing_classes = [cls for cls in expected_classes if cls not in class_dirs]

    if missing_classes:
        raise ValueError(f"Missing classes in '{data_dir}': {missing_classes}")

    for cls in expected_classes:
        class_path = os.path.join(data_dir, cls)
        num_samples = len(os.listdir(class_path))
        if num_samples < min_samples_per_class:
            print(f"Warning: Class '{cls}' in '{data_dir}' has only {num_samples} samples, expected at least {min_samples_per_class}. Proceeding anyway.")
        else:
            print(f"Class '{cls}' in '{data_dir}' meets the sample requirement.")

    print(f"Data check passed for directory: '{data_dir}'")


# Set directories
train_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/train'
val_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/validation'
test_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/test'

# Define expected classes and minimum samples
expected_classes = ['Aero', 'Bacillus', 'Bor', 'E', 'Morganella', 'stapholococcus']
min_samples = 2  # Adjusted to 2 samples per class

# Check data integrity for train, validation, and test sets
check_data_integrity(train_data_dir, expected_classes, min_samples)
check_data_integrity(val_data_dir, expected_classes, min_samples)
check_data_integrity(test_data_dir, expected_classes, min_samples)

# Define image data generators with augmentation for both train and validation sets
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,  # Increased rotation
    width_shift_range=0.3,  # Increased width shift
    height_shift_range=0.3,  # Increased height shift
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    brightness_range=[0.8, 1.2]  # Added brightness adjustment
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load train, validation, and test data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Print class distribution to ensure balanced classes
print(f"Training class distribution: {train_generator.class_indices}")
print(f"Validation class distribution: {val_generator.class_indices}")
print(f"Test class distribution: {test_generator.class_indices}")

# Get class weights for imbalanced classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weights_dict}")

# Build model with additional dropout layers to prevent overfitting
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),  # Added dropout layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Another dropout layer
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer should match number of classes
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for learning rate reduction and early stopping
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

# Train model with class weights to handle class imbalance
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    class_weight=class_weights_dict,  # Adjust for class imbalance
    callbacks=callbacks  # Added callbacks for learning rate reduction and early stopping
)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Confusion matrix and classification report
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion matrix
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred_classes, labels=np.arange(len(expected_classes))))

# Dynamically adjust the labels in the classification report
class_labels = list(test_generator.class_indices.keys())

print("Classification Report")
print(classification_report(y_true, y_pred_classes, target_names=class_labels, labels=np.arange(len(expected_classes)), zero_division=0))
