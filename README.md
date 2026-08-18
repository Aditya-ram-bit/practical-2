 Practical 2: Single Neuron Model using TensorFlow/Keras

 Aim
To implement a small neural network (single hidden layer + output neuron) using TensorFlow/Keras and train it to classify PTAL (Public Transport Accessibility Level) as **Good** or **Poor**, based on walk time to nearest stop and service frequency.


 Theory

Practical 1 used a hand-coded **Perceptron** with a hard step activation and a manual weight-update rule — that only works because AND / the PTAL toy pattern is **linearly separable**. Real-world problems are rarely that clean, so this practical moves to a proper neural network built with a deep learning framework.

**Architecture:**
- **Input layer**: 2 features — `walk_time`, `service_freq` (normalized).
- **Hidden layer**: `Dense(4, activation='relu')` — 4 neurons fully connected to the inputs. ReLU ($f(z) = \max(0, z)$) introduces non-linearity, letting the network learn boundaries a single perceptron cannot.
- **Output layer**: `Dense(1, activation='sigmoid')` — squashes the weighted sum into a probability between 0 and 1. $\text{sigmoid}(z) = \frac{1}{1+e^{-z}}$, suited to binary classification (Good=1 / Poor=0).

**Compilation:**
- **Optimizer — Adam**: adaptive gradient descent; adjusts the learning rate per-parameter using running estimates of the first and second moments of the gradients. Faster and more stable convergence than plain SGD.
- **Loss — binary cross-entropy**: $-[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$, the standard loss for two-class problems; penalizes confident wrong predictions heavily.
- **Metric — accuracy**: fraction of correctly classified samples, tracked per epoch.

**Training — backpropagation:** Unlike the perceptron's manual rule, this network learns via **gradient descent + backpropagation**: the loss is computed on a forward pass, then its gradient is propagated backward through both layers to update every weight, repeated over `epochs`. With very few samples and few epochs, the network is unlikely to fully converge — that's expected and worth observing.

**Perceptron vs. this model:**

| Aspect | Practical 1 (Perceptron) | Practical 2 (Keras NN) |
|---|---|---|
| Layers | Single neuron | Hidden layer (4) + output |
| Activation | Hard step | ReLU + Sigmoid |
| Learning rule | Manual weight update | Backpropagation (Adam) |
| Loss | None (rule-based) | Binary cross-entropy |
| Can learn non-linear patterns | No | Yes |

This is the bridge from a rule-based linear classifier to the gradient-based, non-linear building blocks (hidden layers, activations, backprop) used in modern deep learning.


 2. Single Neuron Model using TensorFlow/Keras
 Domain: PTAL accessibility classification (Good=1 / Poor=0)
import tensorflow as tf
import numpy as np

 Dataset: [walk_time_min (normalized), service_freq_per_hr (normalized)]
 Label 1 = Good PTAL accessibility, 0 = Poor accessibility
X = np.array([
    [0.1, 0.9],
    [0.2, 0.8],
    [0.9, 0.1],
    [0.8, 0.2],
    [0.15, 0.85],
    [0.85, 0.15]
], dtype=np.float32)
y = np.array([[1], [1], [0], [0], [1], [0]], dtype=np.float32)

 Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_dim=2, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')              # Output layer
])

 Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 Predictions before training
labels = {0: 'Poor Access', 1: 'Good Access'}
untrained_predictions = model.predict(X)
print("Before training:")
for i, pred in enumerate(untrained_predictions):
    cls = int(pred >= 0.5)
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => {labels[cls]}")

 Train the model
model.fit(X, y, epochs=10, verbose=2)

 Predictions after training
predictions = model.predict(X)
print("\nAfter training:")
for i, pred in enumerate(predictions):
    cls = int(pred >= 0.5)
    print(f"Input: {X[i]} => Predicted: {round(float(pred), 4)} => {labels[cls]}")
