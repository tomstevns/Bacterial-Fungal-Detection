import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Function to extract epoch information from the log file
def extract_epoch_info(log_file_path):
    epoch_data = []
    epoch_pattern = r'Epoch (\d+)/(\d+)'
    metric_pattern = r'loss: ([\d\.]+) - accuracy: ([\d\.]+) - val_loss: ([\d\.]+) - val_accuracy: ([\d\.]+) - lr: ([\d\.e\-]+)'

    current_epoch = None

    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            metric_match = re.search(metric_pattern, line)
            if metric_match and current_epoch is not None:
                loss = float(metric_match.group(1))
                accuracy = float(metric_match.group(2))
                val_loss = float(metric_match.group(3))
                val_accuracy = float(metric_match.group(4))
                learning_rate = float(metric_match.group(5))
                epoch_data.append([current_epoch, loss, accuracy, val_loss, val_accuracy, learning_rate])

    if not epoch_data:
        print("No epoch information found in the log file. Please check the log format.")
        return None

    df = pd.DataFrame(epoch_data, columns=['Epoch', 'Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy', 'Learning_Rate'])
    return df

# Function to summarize the epoch data
def summarize_epochs(df):
    if df is None or df.empty:
        print("No data available for summary.")
        return None

    first_epoch = df.iloc[0]
    last_epoch = df.iloc[-1]

    summary = {
        "First Epoch": first_epoch.to_dict(),
        "Last Epoch": last_epoch.to_dict(),
        "Max Accuracy": df['Accuracy'].max(),
        "Max Val Accuracy": df['Val_Accuracy'].max(),
        "Min Loss": df['Loss'].min(),
        "Min Val Loss": df['Val_Loss'].min(),
        "Epoch Overfitting Starts": detect_overfitting(df)
    }

    return summary

# Function to detect overfitting
def detect_overfitting(df):
    for i in range(1, len(df)):
        if df['Val_Loss'].iloc[i] > df['Val_Loss'].iloc[i - 1] and df['Loss'].iloc[i] < df['Loss'].iloc[i - 1]:
            return df['Epoch'].iloc[i]
    return None

# Function to plot epoch data with additional insights
def plot_epochs(df):
    if df is None or df.empty:
        print("No data available to plot.")
        return

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Accuracy'], label='Train Accuracy')
    plt.plot(df['Epoch'], df['Val_Accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    overfitting_epoch = detect_overfitting(df)
    if overfitting_epoch:
        plt.suptitle(f'Overfitting detected at Epoch {overfitting_epoch}', fontsize=16, color='red')

    plt.tight_layout()
    plt.show()

# Function to recommend learning rate adjustments
def recommend_learning_rate_adjustment(df):
    if df is None or df.empty:
        return None

    val_loss_improvement = df['Val_Loss'].diff().fillna(0)
    if all(val_loss_improvement.tail(5) >= 0):
        return "Consider lowering the learning rate, as validation loss has plateaued."

    return "Learning rate seems adequate."

# Hyperparameter tuning with improved strategies
def improved_model():
    # Higher L2 regularization and increased dropout
    l2_strength = 0.02  # Increased regularization strength
    dropout_rate = 0.6  # Increased dropout rate

    # EarlyStopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Data augmentation to enhance generalization
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Model creation with L2 regularization and dropout
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3),
                               kernel_regularizer=l2(l2_strength)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_strength)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(6, activation='softmax')  # Adjust based on the number of classes
    ])

    # Compile the model with a lower learning rate for optimization
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, early_stopping, reduce_lr

# Load and process the log file
log_file_path = 'epoch11.log'  # Change to your actual log file path
df = extract_epoch_info(log_file_path)

# Summarize and plot the epochs
summary = summarize_epochs(df)
if summary:
    print("Epoch Summary:", summary)
    learning_rate_recommendation = recommend_learning_rate_adjustment(df)
    print("Learning Rate Recommendation:", learning_rate_recommendation)

plot_epochs(df)
