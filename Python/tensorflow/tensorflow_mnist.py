import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.client import device_lib

def get_gpu_names():
    devices = device_lib.list_local_devices()
    gpu_names = [d.physical_device_desc for d in devices if d.device_type == 'GPU']
    # Extract just the GPU name by splitting the string
    clean_names = []
    for desc in gpu_names:
        # Extract text after 'name: ' and before ', pci bus id'
        start = desc.find('name: ') + len('name: ')
        end = desc.find(',', start)
        clean_names.append(desc[start:end])
    return clean_names

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  # shape: (28, 28, 1)
x_test = x_test[..., tf.newaxis]

# Create a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on GPU (automatically, if available)
model.fit(x_train, y_train,
          epochs=5,
          batch_size=64,
          validation_split=0.1)

gpu_names = get_gpu_names()
gpu_str = ", ".join(gpu_names) if gpu_names else "No GPU detected"

# Evaluate the model and print accuracy and GPU
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy {test_acc:.4f} on GPU {gpu_str}")


