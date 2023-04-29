import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow_addons.optimizers import AdamW
from utils.metrics import KID
from utils import CustomLayer
from utils.setpath import Datapath
from utils.data import load_data

class DiffusionModel(keras.Model):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.path = Datapath()
        config.save_to_json(path=f'{self.path.LOG_CONFIGPATH}/config.json')

        self.normalizer = layers.Normalization()
        self.network = CustomLayer.get_network(config.image_size)
        self.ema_network = keras.models.clone_model(self.network)

        self.compile(optimizer=AdamW(learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay), loss=keras.losses.mean_absolute_error)
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid",image_size=self.config.image_size,kid_image_size=self.config.kid_image_size)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.config.max_signal_rate)
        end_angle = tf.acos(self.config.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, self.config.image_size, self.config.image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.config.batch_size, self.config.image_size, self.config.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.config.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.config.ema * ema_weight + (1 - self.config.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.config.batch_size, self.config.image_size, self.config.image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.config.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=self.config.batch_size, diffusion_steps=self.config.kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, path=''):
        if path=='':
            path = f'{self.path.RESULT_IMGPATH}/{int(time.time())}.png'
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=self.config.plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(path)
        # plt.close()

    def set_callback(self):
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.path.DIFFUSION_PATH,
            save_weights_only=True,
            monitor="val_kid",
            mode="min",
            save_best_only=True,
        )
    def train(self):
        train_dataset, val_dataset = load_data()
        self.normalizer.adapt(train_dataset)
        # run training and plot generated images periodically
        self.history = self.fit(
            train_dataset,
            epochs=self.config.num_epochs,
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.LambdaCallback(on_epoch_end=self.plot_images),
                self.checkpoint_callback,
            ],
        )
        self.save_loss()
    def save_loss(self):
        plt.figure(figsize=(8, 6))
        plt.title('Training Loss',fontsize=20)
        plt.xlabel('Epochs',fontsize=17)
        plt.ylabel('Loss',fontsize=17)
        for key in self.history.history:
            plt.plot(self.history.history[key], label=key)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(f'{self.path.LOG_LOSSPATH}/loss.png')
        for key in self.history.history:
            np.save(arr=self.history.history[key], file=f'{self.path.LOG_LOSSPATH}/{key}.npy')
    def result(self):
        path = f'{self.path.LOG_PREDICTPATH}/predict.png'
        self.load_weights(self.path.DIFFUSION_PATH)
        self.plot_images(path=path)