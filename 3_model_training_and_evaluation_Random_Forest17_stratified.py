import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from imblearn.over_sampling import ADASYN  # Import ADASYN for balancing the classes
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
model_checkpoint_file = 'random_forest_checkpoint.pkl'  # File to store the best model

# Define expected classes and minimum samples
expected_classes = ['Aero', 'Bacillus', 'Bor', 'E', 'Morganella', 'stapholococcus']
min_samples = 2

# Track progress information
start_time = time.time()

# Normalization/Standardization setup
scaler = StandardScaler()

# Function to save the model
def save_model(model, filename=model_checkpoint_file):
    if isinstance(model, (RandomForestClassifier, AdaBoostClassifier, VotingClassifier)):
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
        if os.path.exists(filename):
            print(f"Model file size: {os.path.getsize(filename)} bytes")
    else:
        print("Error: Only supported models can be saved.")

# Function to load the model
def load_model(filename=model_checkpoint_file):
    if os.path.exists(filename):
        print(f"Loading model from {filename}")
        loaded_model = joblib.load(filename)
        if isinstance(loaded_model, (RandomForestClassifier, AdaBoostClassifier, VotingClassifier)):
            return loaded_model
        else:
            print("Error: Loaded file does not contain a valid model.")
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
X_train, y_train = load_data_with_augmentation(train_data_dir, expected_classes, augment=True)

# Standardize intensities
print("Normalizing/Standardizing training data...")
X_train = scaler.fit_transform(X_train)

# Apply ADASYN to balance classes
print("Applying ADASYN to balance classes...")
adasyn = ADASYN(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Further reduce dataset size for faster grid search
X_train_reduced, _, y_train_reduced, _ = train_test_split(X_train_resampled, y_train_resampled, test_size=0.5, random_state=42)

# Standardize the resampled data
X_train_reduced = scaler.fit_transform(X_train_reduced)

# Load test data (without augmentation)
print("Loading and normalizing test data...")
X_test, y_test = load_data_with_augmentation(test_data_dir, expected_classes, augment=False)
X_test = scaler.transform(X_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_reduced)
y_test_encoded = label_encoder.transform(y_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}


# Normalisering/standardisering funktion til intensiteter
def normalize_data(X):
    """
    Normaliser eller standardiser data til en enhedsinterval.
    Dette reducerer variationer i intensitet og sikrer ensartet skala for bedre modeltræning.
    """
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


# Funktion til at indlæse og augmentere data med normalisering
def load_data_with_augmentation(data_dir, expected_classes, img_size=(150, 150), augment=False, normalize=True):
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

                # Normaliser intensiteterne, hvis specificeret
                img_flat = img_resized.flatten()
                if normalize:
                    img_flat = normalize_data(img_flat)

                X.append(img_flat)
                y.append(cls)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(X), np.array(y)


# Load and augment training data with normalization
print("Loading and augmenting (normalized) training data...")
X_train, y_train = load_data_with_augmentation(train_data_dir, expected_classes, augment=True, normalize=True)

# --- Herfra fortsætter koden som tidligere med normaliserede intensiteter ---
# Normalisering/standardisering funktion til intensiteter
def normalize_data(X):
    """
    Normaliser eller standardiser data til en enhedsinterval.
    Dette reducerer variationer i intensitet og sikrer ensartet skala for bedre modeltræning.
    """
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


# Funktion til at indlæse og augmentere data med normalisering
def load_data_with_augmentation(data_dir, expected_classes, img_size=(150, 150), augment=False, normalize=True):
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

                # Normaliser intensiteterne, hvis specificeret
                img_flat = img_resized.flatten()
                if normalize:
                    img_flat = normalize_data(img_flat)

                X.append(img_flat)
                y.append(cls)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(X), np.array(y)


# Load and augment training data with normalization
print("Loading and augmenting (normalized) training data...")
X_train, y_train = load_data_with_augmentation(train_data_dir, expected_classes, augment=True, normalize=True)

# Apply ADASYN to balance classes
print("Applying ADASYN to balance classes...")
adasyn = ADASYN(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Reduce dataset size for faster grid search
X_train_reduced, _, y_train_reduced, _ = train_test_split(X_train_resampled, y_train_resampled, test_size=0.5,
                                                          random_state=42)

# Load test data (without augmentation but with normalization)
print("Loading (normalized) test data...")
X_test, y_test = load_data_with_augmentation(test_data_dir, expected_classes, augment=False, normalize=True)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_reduced)
y_test_encoded = label_encoder.transform(y_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Load existing model checkpoint if available
best_model = load_model()

if best_model is None:
    # Define a parameter grid for RandomForest and AdaBoost hyperparameter tuning
    rf_param_grid = {
        'n_estimators': [3000, 10000, 15000],
        'max_depth': [5, 10, 20],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 2, 3],
        'class_weight': ['balanced']
    }

    abt_param_grid = {
        'algorithm': ['SAMME.R'],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1]
    }

    # Create RandomForestClassifier and AdaBoostClassifier
    rf = RandomForestClassifier(random_state=42)
    base_estimator = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
    abt = AdaBoostClassifier(estimator=base_estimator)

    # Define stratified K-Fold cross-validation
    stratified_kf_rf = StratifiedKFold(n_splits=8)
    stratified_kf_abt = StratifiedKFold(n_splits=8)

    # Use StratifiedKFold in GridSearchCV for both models
    rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=stratified_kf_rf, n_jobs=-1, verbose=4)
    abt_grid_search = GridSearchCV(estimator=abt, param_grid=abt_param_grid, cv=stratified_kf_abt, n_jobs=-1, verbose=4)

    try:
        print("Starting AdaBoosting GridSearchCV...")
        abt_grid_search.fit(X_train_reduced, y_train_encoded)

        print("Starting RandomForest GridSearchCV...")
        rf_grid_search.fit(X_train_reduced, y_train_encoded)

        if rf_grid_search.best_estimator_ and abt_grid_search.best_estimator_:
            best_rf = rf_grid_search.best_estimator_
            best_abt = abt_grid_search.best_estimator_

            best_model = VotingClassifier(estimators=[('rf', best_rf), ('abt', best_abt)], voting='soft')
            best_model.fit(X_train_reduced, y_train_encoded)

            save_model(best_model)
        else:
            print("Error: Could not find a best estimator from the GridSearchCV.")
    except KeyboardInterrupt:
        print("Process interrupted, model checkpoint saved.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
    finally:
        print("GridSearchCV process completed.")

if best_model:
    print("Evaluating model...")
    try:
        y_pred = best_model.predict(X_test)

        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test_encoded)

        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=expected_classes, normalize='true')
        print(f"Normalized Confusion Matrix:\n{cm}")


        def plot_confusion_matrix(cm, classes, normalize=True):
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.show()


        plot_confusion_matrix(cm, expected_classes)

        print("Classification Report:")
        print(classification_report(y_test_labels, y_pred_labels, labels=expected_classes, zero_division=1))

        results_df = pd.DataFrame({
            'Actual': y_test_labels,
            'Predicted': y_pred_labels
        })
        results_df.to_csv('random_forest_results.csv', index=False)
        print(f"Results saved to 'random_forest_results.csv'")

        total_time = time.time() - start_time
        print(f"Total time taken: {total_time // 60:.0f} minutes")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
else:
    print("No model found to evaluate.")
