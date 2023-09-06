#library importing
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets.cifar10 import load_data
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Conv2D, Flatten, LeakyReLU, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

# Load and explore the CIFAR-10 dataset
(train_images, _), (_, _) = load_data()
num_samples = 15  
for i in range(num_samples):
    plt.subplot(4, 10, i + 1)
    plt.axis('off')
    plt.imshow(train_images[i])
plt.show()

# Define the Discriminator Model
def build_discriminator(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample using strides (no max-pooling)
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample using strides
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Downsample using strides
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dropout(0.4))  # Dropout for regularization
    model.add(Dense(1, activation='sigmoid'))  # Output close to 1 means the image is real

    # Compile the model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Model Summary and Visualization
discriminator_model = build_discriminator()
discriminator_model.summary()
plot_model(discriminator_model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)

# Load and preprocess real samples
def load_real_samples():
    (trainX, _), (_, _) = load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5  
    return X

real_samples = load_real_samples()
print(f'Shape of a real sample: {real_samples[0].shape}')

# Generate real samples for training the Discriminator
def generate_real_samples(dataset, n_samples):
    indices = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[indices]
    y = np.ones((n_samples, 1))  # Label real samples as 1
    return X, y

# Generate fake samples for training the Discriminator (usually done by the Generator)
def generate_fake_samples(n_samples):
    X = np.random.rand(32 * 32 * 3 * n_samples)
    X = -1 + X * 2  # Scale to the range [-1, 1]
    X = X.reshape((n_samples, 32, 32, 3))
    y = np.zeros((n_samples, 1))  
    return X, y

# generated fake sample
fake_samples, _ = generate_fake_samples(1)
plt.imshow(fake_samples[0])
plt.show()

# Train the Discriminator
def train_discriminator(d_model, dataset, n_iter=15, n_batch=128):
    half_batch = int(n_batch / 2)
    for epoch in range(n_iter):
        for batch in range(0, dataset.shape[0], n_batch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)

            X_fake, y_fake = generate_fake_samples(half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            print(f'Epoch: {epoch + 1}, Batch: {batch + 1}/{dataset.shape[0]}, Real Loss: {d_loss1}, Fake Loss: {d_loss2}')

# Create the Discriminator Model
discriminator_model = build_discriminator()

# Load Real Samples
real_samples = load_real_samples()

# Train the Discriminator Model
train_discriminator(discriminator_model, real_samples)

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf

# Clear backend session
tf.keras.backend.clear_session()

# Load and preprocess the MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)

# GAN parameters
random_dim = 100
adam = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

# Generator model
generator = Sequential([
    Dense(256, input_dim=random_dim),
    LeakyReLU(0.2),
    BatchNormalization(momentum=0.8),
    Dense(512),
    LeakyReLU(0.2),
    BatchNormalization(momentum=0.8),
    Dense(1024),
    LeakyReLU(0.2),
    BatchNormalization(momentum=0.8),
    Dense(784, activation='tanh'),
    Reshape((28, 28, 1))
])
generator.compile(loss='binary_crossentropy', optimizer=adam)

# Discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(256),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined GAN model
discriminator.trainable = False
gan_input = Input(shape=(random_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=adam)

# Training GAN
def train_gan(epochs=1, batch_size=128):
    batch_count = X_train.shape[0] // batch_size
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        for _ in tqdm(range(batch_count)):
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            generated_images = generator.predict(noise)
            image_indices = np.random.randint(0, X_train.shape[0], size=batch_size)
            image_batch = X_train[image_indices]  # Fetch images using indices

            X = np.concatenate([image_batch, generated_images], axis=0)  # Concatenate along batch axis
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)

            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)

        if (e + 1) % 10 == 0:
            plot_generated_images(e, generator)

def plot_generated_images(epoch, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')

# Train the GAN with fine-tuned parameters
train_gan(epochs=200, batch_size=128)

