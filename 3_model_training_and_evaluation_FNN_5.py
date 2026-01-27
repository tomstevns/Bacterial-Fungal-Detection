import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter


# Funktion til at tælle billeder i hver klasse
def count_images_per_class(directory):
    class_counts = Counter()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                class_name = os.path.basename(root)
                class_counts[class_name] += 1
    return class_counts


# Funktion til at balancere klasser med data augmentation
def balance_classes(train_dir, target_class_size=10):
    # Opsæt en data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    class_counts = count_images_per_class(train_dir)

    for class_name, count in class_counts.items():
        if count < target_class_size:
            # Skab flere billeder ved augmentation
            class_dir = os.path.join(train_dir, class_name)
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(".jpg")]

            print(f"Augmenting class '{class_name}' with {target_class_size - count} additional images.")

            for i in range(target_class_size - count):
                img_path = images[i % len(images)]
                img = tf.keras.preprocessing.image.load_img(img_path)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array.reshape((1,) + img_array.shape)

                # Gem de nye augmented billeder
                save_prefix = f'aug_{i}_{os.path.basename(img_path)}'
                save_dir = os.path.join(class_dir, save_prefix)
                datagen.flow(img_array, batch_size=1, save_to_dir=class_dir, save_prefix=save_prefix, save_format='jpg')


# Brug funktionen til at balancere dine klasser
train_dir = 'data/train'
balance_classes(train_dir, target_class_size=10)
