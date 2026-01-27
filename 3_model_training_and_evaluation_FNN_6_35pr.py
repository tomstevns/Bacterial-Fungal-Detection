import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pandas as pd
import itertools

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.0005, 0.0001],
    'dropout_rate': [0.4, 0.6],  # Increased dropout rate
    'filters': [32, 64],
    'rotation_range': [40, 50],
    'zoom_range': [0.2, 0.3],
    'epochs': [10, 15]
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
check_data_integrity(train_data_dir, expected_classes, min_samples)
check_data_integrity(val_data_dir, expected_classes, min_samples)
check_data_integrity(test_data_dir, expected_classes, min_samples)

# Function to create the model based on current hyperparameters with L2 regularization
def create_model(learning_rate, dropout_rate, filters, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(150, 150, 3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Increased L2 regularization
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(dropout_rate),  # Dropout to prevent overfitting
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Dense units with L2
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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

    # Define image data generators with current augmentation settings
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=rotation_range,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=zoom_range,
        shear_range=0.2,  # Added shear transformation for augmentation
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load data with adjusted batch size
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_data_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False
    )

    # Get number of classes from train_generator
    num_classes = train_generator.num_classes

    # Get class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes),
                                         y=train_generator.classes)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Create and compile the model
    model = create_model(learning_rate, dropout_rate, filters, num_classes)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001),  # Increase patience to 4
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),  # Stop if no improvement
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)  # Save best model
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Save history for analysis
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"history_{learning_rate}_{dropout_rate}_{filters}.csv", index=False)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    val_loss = min(history.history['val_loss'])  # Best validation loss

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
    pd.DataFrame(results_log).to_csv('hyperparameter_results.csv', index=False)

# Once all experiments are done, review the CSV file to select the best parameters.
