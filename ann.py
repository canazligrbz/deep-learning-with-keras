# Import necessary libraries
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize images
# Flatten 28x28 -> 784 and scale pixel values from [0,255] to [0,1]
x_train = x_train.reshape((x_train.shape[0], 28*28)).astype("float32") / 255
x_test  = x_test.reshape((x_test.shape[0], 28*28)).astype("float32") / 255

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Build ANN model
model = Sequential()

# First hidden layer: 512 neurons, ReLU activation
model.add(Dense(512, activation="relu", input_shape=(28*28,)))

# Second hidden layer: 256 neurons, tanh activation
model.add(Dense(256, activation="tanh"))

# Output layer: 10 neurons (for 10 classes), softmax activation
model.add(Dense(10, activation="softmax"))

# Print model summary
model.summary()

# Compile the model
# Adam optimizer, categorical crossentropy loss, accuracy metric
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Model checkpoint callback
checkpoint = ModelCheckpoint("ann_best_model.h5", save_best_only=True, monitor="val_loss")

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=60,
                    validation_split=0.2,
                    callbacks=[early_stopping, checkpoint])

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test acc: {test_acc}, Test loss: {test_loss}")

# Plot accuracy
plt.figure()
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("ANN Accuracy on MNIST Data Set") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.legend() 
plt.grid(True) 
plt.show()

# Plot loss
plt.figure()
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.title("ANN Loss on MNIST Data Set") 
plt.xlabel("Epochs") 
plt.ylabel("Loss") 
plt.legend()
plt.grid(True) 
plt.show()

# Save the final model
model.save("final_mnist_ann_model.h5")

# Load the saved model
loaded_model = load_model("final_mnist_ann_model.h5")

# Evaluate loaded model
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Loaded Model Result -> Test acc: {test_acc}, Test loss: {test_loss}")


















