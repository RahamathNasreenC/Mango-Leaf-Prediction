# cnn_model_transfer.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import os

#  Paths
train_dir = 'data/train'
test_dir = 'data/test'


# Data Preprocessing with augmentation

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load data
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=8,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=8,
    class_mode='binary'
)

# Save class indices for mapping
class_indices = train_set.class_indices
class_names = [None]*len(class_indices)
for name, index in class_indices.items():
    class_names[index] = name

print("Class indices:", class_indices)
print("Class names:", class_names)


# Load pre-trained MobileNetV2

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_model.trainable = False  # freeze base layers

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)


#  Compile model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train model

history = model.fit(
    train_set,
    validation_data=test_set,
    epochs=15
)


# Save model

os.makedirs('models', exist_ok=True)
model.save('models/leaf_model_transfer.h5')
print("✅ Model trained and saved as 'models/leaf_model_transfer.h5'")


# Evaluate on test set

test_loss, test_acc = model.evaluate(test_set)
print(f"🎯 Test Accuracy: {test_acc*100:.2f}%")


# Plot training graphs

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()
