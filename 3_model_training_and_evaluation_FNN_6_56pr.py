import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'dropout_rate': [0.3, 0.4, 0.5],
    'neurons': [64, 128, 256],
    'epochs': [25, 50, 100]
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

# Image augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.2,
    shear_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Create CNN model with regularization and dropout
def create_cnn_model(learning_rate, dropout_rate, neurons, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(neurons, (3, 3), activation='relu', input_shape=(150, 150, 3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(neurons, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Learning rate scheduler function
def lr_scheduler(epoch, lr):
    return lr * np.exp(-0.05 * epoch)

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
    learning_rate, dropout_rate, neurons, epochs = params

    # Load data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir, target_size=(150, 150), batch_size=16, class_mode='categorical', shuffle=False
    )

    # Get number of classes from train_generator
    num_classes = train_generator.num_classes

    # Get class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes),
                                         y=train_generator.classes)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Create and compile the model
    model = create_cnn_model(learning_rate, dropout_rate, neurons, num_classes)

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
        LearningRateScheduler(lr_scheduler)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator)
    val_loss = min(history.history['val_loss'])  # Best validation loss

    # Log the results
    log_results(dict(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        neurons=neurons,
        epochs=epochs
    ), test_acc, val_loss)

    # Generate predictions for confusion matrix
    y_pred = model.predict(test_generator)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = test_generator.classes

    # Compute normalized confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, normalize='true')

    # Plot normalized confusion matrix
    def plot_confusion_matrix(cm, classes):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    # Call the plot function for each epoch
    #plot_confusion_matrix(cm, classes=test_generator.class_indices.keys())

# Call the plot function for last epoch
plot_confusion_matrix(cm, classes=test_generator.class_indices.keys())
# Save the results to a CSV file for review
pd.DataFrame(results_log).to_csv('hyperparameter_results.csv', index=False)
