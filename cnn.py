from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, 
                                     BatchNormalization, GlobalAveragePooling2D, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# Load dataset

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype("float32") / 255
x_test  = x_test.astype("float32") / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)


# Data Augmentation

# Create new variations of training images to reduce overfitting
datagen = ImageDataGenerator(
    rotation_range=15,     # random rotations
    width_shift_range=0.1, # horizontal shifts
    height_shift_range=0.1,# vertical shifts
    horizontal_flip=True   # random horizontal flips
)
datagen.fit(x_train)


# Model

model = Sequential()

# Block 1: Learn low-level features (edges, textures)
model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))   # downsampling
model.add(Dropout(0.25))         # prevent overfitting

# Block 2: Learn more complex features (shapes, parts of objects)
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Block 3: Learn high-level features (object parts, patterns)
model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# Classification layers
model.add(GlobalAveragePooling2D())  # convert feature maps to vector
model.add(Dense(128, activation="relu")) 
model.add(Dropout(0.5))              # stronger regularization
model.add(Dense(10, activation="softmax"))  # final class probabilities


"""
Conv + ReLU   : Görselden özellik çıkarır (kenar, desen, obje parçaları).
Pooling       : Boyutu küçültür, en önemli bilgileri saklar.
Dropout       : Aşırı öğrenmeyi (overfitting) engeller.
Flatten / GAP : 2D özellik haritalarını tek boyutlu vektör haline getirir.
Dense (ReLU)  : Özellikleri birleştirip daha anlamlı temsiller oluşturur.
Dense (Softmax): Çıkış katmanı; hangi sınıfa ait olduğunu olasılık ile tahmin eder.
"""


# Compile

model.compile(
    optimizer=Adam(learning_rate=1e-3),   # optimizer with learning rate
    loss=CategoricalCrossentropy(label_smoothing=0.1), # loss with label smoothing
    metrics=["accuracy"]
)


# Callbacks

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"), # stop if no improvement
    ReduceLROnPlateau(factor=0.5, patience=5, monitor="val_loss"),                 # reduce LR on plateau
    ModelCheckpoint("best_cnn.h5", save_best_only=True, monitor="val_accuracy")    # save best model
]


# Training

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64), # use augmented data
    validation_data=(x_test, y_test),              # validate on test data
    epochs=50,
    callbacks=callbacks
)


# Evaluation

y_pred= model.predict(x_test)
y_pred_class= np.argmax(y_pred, axis=1)
y_true= np.argmax(y_test, axis=1)

report= classification_report(y_true, y_pred_class)
print(report)


# Plots

# Plot training vs validation loss
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss") 
plt.xlabel("Epochs") 
plt.ylabel("Loss") 
plt.legend() 
plt.grid(True) 
plt.show()

# Plot training vs validation accuracy
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.legend()
plt.grid(True) 
plt.show()




