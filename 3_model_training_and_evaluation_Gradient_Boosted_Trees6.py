import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from imblearn.over_sampling import SMOTE
import itertools
import matplotlib.pyplot as plt
import joblib  # For saving and loading models
import time  # To track time for qualitative progress

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Set directories
train_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/train'
test_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/test'
model_checkpoint_file = 'gradient_boosting_checkpoint6.pkl'  # File to store model

# Define expected classes and minimum samples
expected_classes = ['Aero', 'Bacillus', 'Bor', 'E', 'Morganella', 'stapholococcus']
min_samples = 2

# Track progress information
start_time = time.time()

# Function to save the model
def save_model(model, filename=model_checkpoint_file):
    if isinstance(model, GradientBoostingClassifier):
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
        if os.path.exists(filename):
            print(f"Model file size: {os.path.getsize(filename)} bytes")
    else:
        print("Error: Only GradientBoostingClassifier models can be saved.")

# Function to load the model
def load_model(filename=model_checkpoint_file):
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        loaded_model = joblib.load(filename)
        if isinstance(loaded_model, GradientBoostingClassifier):
            return loaded_model
        else:
            print("Error: Loaded file does not contain a valid GradientBoostingClassifier model.")
    return None

# Faster data augmentation by reducing the number of transformations
def augment_image(img, img_size, is_bor=False, is_staph=False):
    img_resized = resize(img, img_size)  # Resize back to the desired size
    img_brightened = exposure.adjust_gamma(img_resized, gamma=random.uniform(0.9, 1.1))  # Minor brightness change
    return img_brightened

# Load images with reduced augmentation to speed up
def load_data_with_augmentation(data_dir, expected_classes, img_size=(150, 150), augment=False):
    X = []
    y = []
    for cls in expected_classes:
        class_dir = os.path.join(data_dir, cls)
        if not os.path.exists(class_dir):
            print(f"Warning: Class '{cls}' not found in '{data_dir}'. Skipping.")
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = imread(img_path)
                img_resized = resize(img, img_size, anti_aliasing=True)
                if augment:
                    is_bor = cls == 'Bor'
                    is_staph = cls == 'stapholococcus'
                    img_resized = augment_image(img_resized, img_size, is_bor=is_bor, is_staph=is_staph)
                X.append(img_resized.flatten())
                y.append(cls)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(X), np.array(y)

# Load and augment training data
print("Loading and augmenting training data...")
X_train, y_train = load_data_with_augmentation(train_data_dir, expected_classes, augment=False)

# Apply SMOTE to balance classes
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Reduce dataset size for faster grid search
X_train_reduced, _, y_train_reduced, _ = train_test_split(X_train_resampled, y_train_resampled, test_size=0.5, random_state=42)

# Load test data (without augmentation)
print("Loading test data...")
X_test, y_test = load_data_with_augmentation(test_data_dir, expected_classes, augment=False)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_reduced)
y_test_encoded = label_encoder.transform(y_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Load existing model checkpoint if available
best_gbt = load_model()

if best_gbt is None:
    # Define a smaller parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [300, 500],  # Fewer n_estimators to reduce time
        'max_depth': [3, 5],  # Use simpler trees for faster training
        'learning_rate': [0.01, 0.05],  # Focus on smaller learning rates
        'subsample': [0.7],  # Fix subsample to avoid too many combinations
        'min_samples_split': [5, 10],  # Reduce overfitting with larger splits
        'min_samples_leaf': [2, 3]  # Generalization by increasing min_samples_leaf
    }

    # Create a GradientBoostingClassifier with warm_start enabled
    gbt = GradientBoostingClassifier(random_state=42, warm_start=True)

    # Use 3-fold cross-validation to reduce training time
    grid_search = GridSearchCV(estimator=gbt, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)

    # Fit the model
    try:
        print("Starting GridSearchCV...")
        grid_search.fit(X_train_reduced, y_train_encoded)
        best_gbt = grid_search.best_estimator_
        save_model(best_gbt)  # Save the best model found

        # Get the results of all fits
        results = pd.DataFrame(grid_search.cv_results_)

        # Save results
        results.to_csv('grid_search_fit_results_reduced.csv', index=False)
        print("All fit results saved to 'grid_search_fit_results_reduced.csv'")

    except KeyboardInterrupt:
        if hasattr(grid_search, 'best_estimator_'):
            save_model(grid_search.best_estimator_)  # Save model when interrupted
        print("Process interrupted, model checkpoint saved.")

# Ensure the model is loaded correctly and proceed to evaluation
if best_gbt:
    print("Evaluating model...")
    try:
        # Evaluate the best model on the test data
        y_pred = best_gbt.predict(X_test)

        # Decode labels back to original class names
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test_encoded)

        # Compute confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=expected_classes)
        print(f"Confusion Matrix:\n{cm}")

        # Plot confusion matrix
        def plot_confusion_matrix(cm, classes):
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()

        # Call the plot function for the confusion matrix
        plot_confusion_matrix(cm, expected_classes)

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test_labels, y_pred_labels, labels=expected_classes, zero_division=1))

        # Save the results to a CSV file for review
        results_df = pd.DataFrame({
            'Actual': y_test_labels,
            'Predicted': y_pred_labels
        })
        results_df.to_csv('gradient_boosting_results.csv', index=False)
        if os.path.exists('gradient_boosting_results.csv'):
            print(f"Results file size: {os.path.getsize('gradient_boosting_results.csv')} bytes")
        print("All results saved to 'gradient_boosting_results.csv'")

        # Qualitative progress tracking
        total_time = time.time() - start_time
        print(f"Total time taken: {total_time // 60:.0f} minutes")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
else:
    print("No model found to evaluate.")
