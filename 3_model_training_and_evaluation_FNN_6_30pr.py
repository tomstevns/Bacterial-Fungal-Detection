import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd  # For logging results
import itertools  # For generating hyperparameter combinations

# Define the parameter grid for hyperparameter tuning with further reduced learning rates and higher dropout rates
param_grid = {
    'learning_rate': [0.000001, 0.0000001],  # Further reduced learning rates for more precise tuning
    'dropout_rate': [0.8, 0.9],  # Higher dropout rates to prevent overfitting
    'filters': [8],  # Further simplified model to reduce overfitting
    'rotation_range': [70],  # Aggressive data augmentation
    'zoom_range': [0.4],  # Continued aggressive data augmentation
    'epochs': [40, 50]  # Extended epochs to allow fine-tuning over longer periods
}

# Set directories
train_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/train'
val_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/validation'
test_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/test'

# Define expected classes and minimum samples
expected_classes = ['Aero', 'Bacillus', 'Bor', 'E', 'Morganella', 'stapholococcus']
min_samples = 2

# Check data integrity function
def check_data_integrity(data_dir, expected_classes, min_samples_per_class=1):
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
            print(f"Warning: Class '{cls}' has only {num_samples} samples, expected at least {min_samples_per_class}.")
        else:
            print(f"Class '{cls}' meets the sample requirement.")
    print(f"Data check passed for directory: '{data_dir}'")

# Check data integrity for all sets
print("Checking data integrity...")
check_data_integrity(train_data_dir, expected_classes, min_samples)
check_data_integrity(val_data_dir, expected_classes, min_samples)
check_data_integrity(test_data_dir, expected_classes, min_samples)

print("Data integrity check complete.")

# Function to create the model based on current hyperparameters with updated architecture
def create_model(learning_rate, dropout_rate, filters, num_classes):
    print(f"Creating model with learning rate: {learning_rate}, dropout: {dropout_rate}, filters: {filters}")
    model = tf.keras.models.Sequential([
        # Increased L2 regularization to combat overfitting
        tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(150, 150, 3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.03)),  # Further increased L2 regularization
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.03)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(filters * 4, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.03)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(dropout_rate),  # Increased dropout to prevent overfitting
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.03)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model created and compiled.")
    return model

# Logging function
results_log = []

def log_results(params, test_acc, val_loss):
    log_entry = params.copy()
    log_entry['test_accuracy'] = test_acc
    log_entry['val_loss'] = val_loss
    results_log.append(log_entry)

# Iterate over all combinations of hyperparameters
param_combinations = list(itertools.product(*param_grid.values()))

for params in param_combinations:
    # Unpack the parameters
    learning_rate, dropout_rate, filters, rotation_range, zoom_range, epochs = params
    print(f"\nRunning training with params: {params}")

    # Define image data generators with current augmentation settings
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=rotation_range,
        width_shift_range=0.4,  # Aggressive shifts to prevent overfitting
        height_shift_range=0.4,
        zoom_range=zoom_range,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3]  # More aggressive brightness augmentation
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load data with adjusted batch size
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical'
    )
    print("Loading validation data...")
    val_generator = val_datagen.flow_from_directory(
        val_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical'
    )
    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        test_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical', shuffle=False
    )

    # Get number of classes from train_generator
    num_classes = train_generator.num_classes
    print(f"Number of classes: {num_classes}")

    # Get class weights
    print("Computing class weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes),
                                         y=train_generator.classes)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights: {class_weights_dict}")

    # Create and compile the model
    model = create_model(learning_rate, dropout_rate, filters, num_classes)

    # Callbacks for reducing learning rate on plateau and early stopping
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001),
        EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)  # Early stopping set to patience 1
    ]

    # Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(test_generator)
    val_loss = min(history.history['val_loss'])
    print(f"Test accuracy: {test_acc}, Validation loss: {val_loss}")

    # Log the results
    log_results(dict(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        filters=filters,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        epochs=epochs
    ), test_acc, val_loss)

    # Save the results to a CSV file for review
    print("Saving results to CSV...")
    pd.DataFrame(results_log).to_csv('hyperparameter_results.csv', index=False)

print("All experiments completed.")
