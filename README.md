# FractalNet in TensorFlow

FractalNet is a deep neural network architecture that incorporates a recursive fractal design pattern to improve convergence and performance without the use of shortcut (residual) connections. This network design creates multiple sub-networks of different depths, allowing the model to naturally balance between depth and width, promoting diversity in feature learning and regularization.

This implementation uses TensorFlow and Keras to create FractalNet, which is adaptable for various image classification tasks such as CIFAR-10.

## Key Features of this Implementation:
- **Fractal Blocks**: The core of FractalNet is the recursive fractal expansion of convolutional layers.
- **Dropout and Batch Normalization**: Each convolutional layer is followed by batch normalization and optional dropout for improved regularization and faster convergence.
- **Adjustable Depth and Filters**: You can configure the depth of fractal expansion and the number of filters in the convolutional layers.
- **Pooling Layers**: Between fractal blocks, max-pooling layers downsample the feature maps, progressively reducing their size.

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Model Structure](#model-structure)
4. [Parameters](#parameters)
5. [Training](#training)
6. [Citing FractalNet](#citing-fractalnet)

---

## Installation

To use this code, you’ll need the following:
1. Python 3.x
2. TensorFlow 2.x or higher
3. Keras (included with TensorFlow 2.x)

You can install TensorFlow via pip if you don’t have it installed:

```bash
pip install tensorflow
```

---

## Usage

### Building the FractalNet Model

You can create the FractalNet model by calling the `build_fractalnet` function, where you can specify parameters like input shape, number of classes, depth, initial number of filters, and dropout rate.

```python
from fractalnet import build_fractalnet

# Define input shape and number of classes (for example, CIFAR-10)
input_shape = (32, 32, 3)  # 32x32 RGB images
num_classes = 10           # Number of classes

# Build the FractalNet model
model = build_fractalnet(input_shape, num_classes, initial_filters=32, depth=3, dropout_rate=0.3)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
```

### Training

Once you’ve built the model, you can train it on your dataset as usual using the `model.fit()` method.

```python
# Assuming you have your training data prepared as (X_train, y_train) and (X_test, y_test)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)
```

---

## Model Structure

The architecture of FractalNet in this implementation consists of:
1. **Initial Convolution Layer**: A Conv2D layer at the input level for initial feature extraction.
2. **Fractal Blocks**: These blocks recursively combine two paths, each path being a sub-network at a lower depth.
   - Each path contains Conv2D layers, batch normalization, and dropout (if enabled).
3. **Max Pooling Layers**: After each fractal block, a max-pooling layer downsamples the feature maps.
4. **Fully Connected Layers**: After the fractal blocks, the network flattens the feature maps and connects to dense layers for classification.

### Parameters

- **`input_shape`**: Shape of the input data (e.g., `(32, 32, 3)` for CIFAR-10).
- **`num_classes`**: Number of output classes (e.g., 10 for CIFAR-10).
- **`initial_filters`**: Number of filters in the first convolutional layer, which doubles after each fractal block.
- **`depth`**: The depth of the fractal expansion. Higher depth creates more recursive fractal blocks.
- **`dropout_rate`**: Dropout rate applied after each convolutional block.

---
