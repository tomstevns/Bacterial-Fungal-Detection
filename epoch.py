import re
import pandas as pd
import matplotlib.pyplot as plt


# Define a function to extract epoch information from the log file
def extract_epoch_info(log_file_path):
    epoch_data = []
    # Pattern for matching 'Epoch X/Y' lines
    epoch_pattern = r'Epoch (\d+)/(\d+)'
    # Pattern for matching lines with loss, accuracy, val_loss, val_accuracy, lr
    metric_pattern = r'loss: ([\d\.]+) - accuracy: ([\d\.]+) - val_loss: ([\d\.]+) - val_accuracy: ([\d\.]+) - lr: ([\d\.e\-]+)'

    current_epoch = None

    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Check for epoch lines first
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))  # Capture current epoch number
                continue

            # Check for metric lines
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

    # Return a DataFrame with the extracted data
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
    # Overfitting is detected when validation loss starts increasing while training loss decreases
    for i in range(1, len(df)):
        if df['Val_Loss'].iloc[i] > df['Val_Loss'].iloc[i - 1] and df['Loss'].iloc[i] < df['Loss'].iloc[i - 1]:
            return df['Epoch'].iloc[i]
    return None


# Function to plot epoch data with additional insights
def plot_epochs(df):
    if df is None or df.empty:
        print("No data available to plot.")
        return

    # Create a subplot for loss and accuracy
    plt.figure(figsize=(10, 5))

    # Plot loss per epoch
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy per epoch
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Accuracy'], label='Train Accuracy')
    plt.plot(df['Epoch'], df['Val_Accuracy'], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Highlight overfitting epoch if detected
    overfitting_epoch = detect_overfitting(df)
    if overfitting_epoch:
        plt.suptitle(f'Overfitting detected at Epoch {overfitting_epoch}', fontsize=16, color='red')

    plt.tight_layout()
    plt.show()


# Function to recommend learning rate adjustments
def recommend_learning_rate_adjustment(df):
    if df is None or df.empty:
        return None

    # Check for a plateau in validation loss and accuracy
    val_loss_improvement = df['Val_Loss'].diff().fillna(0)
    if all(val_loss_improvement.tail(5) >= 0):
        return "Consider lowering the learning rate, as validation loss has plateaued."

    return "Learning rate seems adequate."


# Load and process the log file
log_file_path = 'epoch54.log'  # Change to your actual log file path
df = extract_epoch_info(log_file_path)

# Summarize and plot the epochs
summary = summarize_epochs(df)
if summary:
    print("Epoch Summary:", summary)

    # Provide learning rate adjustment recommendations
    learning_rate_recommendation = recommend_learning_rate_adjustment(df)
    print("Learning Rate Recommendation:", learning_rate_recommendation)

plot_epochs(df)
