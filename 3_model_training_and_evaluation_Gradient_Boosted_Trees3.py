import os
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage import exposure
from imblearn.over_sampling import SMOTE
import itertools
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import time
from sklearn.model_selection import ParameterGrid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Set directories
train_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/train'
test_data_dir = 'C:/Users/tom.zbc/PycharmProjects/fungis_bacterias_project/data/test'

# Define expected classes and minimum samples
expected_classes = ['Aero', 'Bacillus', 'Bor', 'E', 'Morganella', 'stapholococcus']
min_samples = 2

# Aggressive augmentation for Bor and stapholococcus
def augment_image(img, img_size, is_bor=False, is_staph=False):
    img_rotated = rotate(img, angle=random.uniform(-40, 40), mode='wrap')  # Rotate more aggressively
    img_brightened = exposure.adjust_gamma(img_rotated, gamma=random.uniform(0.3, 2.5))  # Strong brightness change
    img_resized = resize(img_brightened, img_size)  # Resize back to the desired size
    return img_resized

# Function to load images and labels with aggressive augmentation for Bor and stapholococcus
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
X_train, y_train = load_data_with_augmentation(train_data_dir, expected_classes, augment=True)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Load test data (without augmentation)
X_test, y_test = load_data_with_augmentation(test_data_dir, expected_classes, augment=False)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_resampled)
y_test_encoded = label_encoder.transform(y_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
class_weights_dict[expected_classes.index('Bor')] = 4.5
class_weights_dict[expected_classes.index('E')] = 5.0
class_weights_dict[expected_classes.index('stapholococcus')] = 3.0

# Define the parameter grid manually
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create a parameter grid
param_combinations = list(ParameterGrid(param_grid))

# Check if checkpoint exists
checkpoint_file = 'gradient_boosting_checkpoint6.pkl'
completed_combinations = []

if os.path.exists(checkpoint_file):
    print("Loading checkpoint...")
    with open(checkpoint_file, 'rb') as f:
        completed_combinations = joblib.load(f)
else:
    completed_combinations = []

# Create GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state=42)

# Track training time
start_time = time.time()

# Progress bar for total combinations
with tqdm(total=len(param_combinations), desc="Parameter Combinations") as pbar:
    for i, params in enumerate(param_combinations):
        # Skip combinations already completed
        if i in completed_combinations:
            pbar.update(1)
            continue

        gbt.set_params(**params)
        gbt.fit(X_train_resampled, y_train_encoded)

        # Evaluate the model on test data
        y_pred = gbt.predict(X_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test_encoded)

        # Compute confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=expected_classes)
        print(f"Confusion Matrix for combination {i}:\n{cm}")

        # Save progress after each iteration
        completed_combinations.append(i)
        with open(checkpoint_file, 'wb') as f:
            joblib.dump(completed_combinations, f)

        # Save results to CSV
        results_df = pd.DataFrame({
            'Actual': y_test_labels,
            'Predicted': y_pred_labels
        })
        results_df.to_csv(f'gradient_boosting_results_{i}.csv', index=False)

        pbar.update(1)

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")
