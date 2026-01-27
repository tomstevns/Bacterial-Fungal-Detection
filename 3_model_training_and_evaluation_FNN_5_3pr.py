import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

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
            class_dir = os.path.join(train_dir, class_name)
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(".jpg")]

            print(f"Augmenting class '{class_name}' with {target_class_size - count} additional images.")

            for i in range(target_class_size - count):
                img_path = images[i % len(images)]
                img = tf.keras.preprocessing.image.load_img(img_path)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array.reshape((1,) + img_array.shape)

                save_prefix = f'aug_{i}_{os.path.basename(img_path)}'
                save_dir = os.path.join(class_dir, save_prefix)
                datagen.flow(img_array, batch_size=1, save_to_dir=class_dir, save_prefix=save_prefix, save_format='jpg')

# Brug funktionen til at balancere dine klasser
train_dir = 'data/train'
balance_classes(train_dir, target_class_size=10)

# Opsæt ImageDataGenerators til træning, validering og test
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

# Byg modellen
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout for regularization
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# Ændre den sidste dense-lag til at have 6 neuroner
model.add(layers.Dense(6, activation='softmax'))


# Opsæt optimizer og læringsrate
optimizer = optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Tilføj Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Træn modellen
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    callbacks=[early_stopping]
)

# Evaluér modellen på testdata
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Gem modellen
model.save('model_fnn_augmented_fixed.h5')
