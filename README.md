# practical-2
# Single Neuron Model using TensorFlow/Keras — Explanation

This notebook implements a single neuron model (a small neural network) using TensorFlow/Keras, and trains it to approximate the AND logic gate. Here's what's happening step by step:

## 1. Importing Libraries

```python
import tensorflow as tf
import numpy as np
```

- **tensorflow**: Used to build and train neural networks.
- **numpy**: Used for numerical computations and array manipulations.

## 2. Defining the Dataset

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
```

`X` is the input dataset representing the AND gate inputs. `y` is the output label, where the output is 1 only when both inputs are 1, as per the truth table of the AND gate.

## 3. Building the Neural Network Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')              # Output layer
])
```

- **Sequential**: Creates a linear stack of layers.
- **First Layer (Hidden Layer)**: `Dense(4)` — a fully connected layer with 4 neurons. `input_dim=2` accepts 2 input features (each input sample has two values). `activation='relu'` introduces non-linearity.
- **Second Layer (Output Layer)**: `Dense(1)` outputs a single value. `activation='sigmoid'` squashes the output between 0 and 1, suitable for binary classification.

## 4. Compiling the Model

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **optimizer='adam'**: An efficient optimization algorithm used to update weights.
- **loss='binary_crossentropy'**: The loss function suited for binary classification problems.
- **metrics=['accuracy']**: Tracks accuracy during training.

## 5. Predictions Before Training

```python
untrained_predictions = model.predict(X)
for i, pred in enumerate(untrained_predictions):
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0)}")
```

Before any training, the model's weights are randomly initialized, so its predictions carry no real pattern — they hover close to 0.5, reflecting pure guesswork.

## 6. Training the Model

```python
model.fit(X, y, epochs=10, verbose=2)
```

The model is trained for 10 epochs, adjusting its internal weights each pass to reduce the loss and better fit the AND pattern.

## 7. Evaluating and Making Predictions

```python
predictions = model.predict(X)
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => Class: {int(pred >= 0.5)}")
```

Predicts outputs for all input combinations, rounding the sigmoid output and applying a 0.5 threshold to classify each result as 0 or 1.

## 8. The Output

**Before training:**
```
Input: [0. 0.] => Predicted: 0.5    => Class: 1
Input: [0. 1.] => Predicted: 0.5783 => Class: 1
Input: [1. 0.] => Predicted: 0.5357 => Class: 1
Input: [1. 1.] => Predicted: 0.5902 => Class: 1
```

**After 10 epochs of training:**
```
Input: [0. 0.] => Predicted: 0.4975 => Class: 0
Input: [0. 1.] => Predicted: 0.5704 => Class: 1
Input: [1. 0.] => Predicted: 0.5306 => Class: 1
Input: [1. 1.] => Predicted: 0.582  => Class: 1
```

Training loss dropped only slightly (from ~0.713 to ~0.708) and accuracy stayed at 50%. This confirms:

- Before training, the model behaves essentially randomly, since its weights start with no knowledge of the AND pattern.
- After only 10 epochs, the model has barely begun to learn — it correctly classifies `[0,0]` now, but still misclassifies `[0,1]` and `[1,0]` as 1 instead of 0.
- This is expected: with such a tiny dataset (only 4 samples) and just 10 epochs, the network hasn't had enough iterations to converge. Unlike the earlier perceptron (which used a hard-coded learning rule for a linearly separable problem), this model relies on gradient descent via backpropagation, which typically needs many more epochs to fully converge on such a small toy dataset.

Bigger picture: this notebook shows the same AND gate problem tackled with a real neural network framework instead of a manual perceptron — introducing hidden layers, non-linear activations (ReLU, Sigmoid), and gradient-based optimization (Adam + binary cross-entropy), which are the actual building blocks used in modern deep learning models.
