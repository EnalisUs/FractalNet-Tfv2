import tensorflow as tf
from tensorflow.keras import layers, models

# Define fractal block
def fractal_block(input_tensor, filters, depth, dropout_rate):
    if depth == 1:
        # Base case: a simple conv block
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
        return x
    else:
        # Recursive case: fractal pattern
        x1 = fractal_block(input_tensor, filters, depth - 1, dropout_rate)
        x2 = fractal_block(input_tensor, filters, depth - 1, dropout_rate)
        
        # Combine both paths (fractal structure)
        combined = layers.Add()([x1, x2])
        return combined

# Build the FractalNet model
def build_fractalnet(input_shape, num_classes, initial_filters=32, depth=3, dropout_rate=0.3):
    inputs = tf.keras.Input(shape=input_shape)

    # Initial conv layer
    x = layers.Conv2D(initial_filters, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    # Fractal blocks
    filters = initial_filters
    for i in range(depth):
        x = fractal_block(x, filters, depth-i, dropout_rate)
        filters *= 2
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = models.Model(inputs, outputs)
    return model

# Set parameters
input_shape = (32, 32, 3)  # For example, CIFAR-10 dataset
num_classes = 10           # Number of classes for classification

# Create the FractalNet model
model = build_fractalnet(input_shape, num_classes, initial_filters=32, depth=3, dropout_rate=0.3)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
