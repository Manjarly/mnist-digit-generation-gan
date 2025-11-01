# PyTorch Generative Adversarial Network (GAN) for MNIST

This repository contains a Jupyter Notebook (`GAN_for_MNIST.ipynb`) that implements a simple Generative Adversarial Network (GAN) from scratch. The network is trained on the MNIST dataset to generate new, synthetic images of handwritten digits.

## 🚀 Project Overview

The notebook demonstrates the full implementation of a GAN, which consists of two competing neural networks:
1.  **A Generator ($G$):** Tries to create realistic images of handwritten digits from random noise.
2.  **A Discriminator ($D$):** Tries to distinguish between real images (from the MNIST dataset) and fake images (created by the generator).

Through this adversarial process, the Generator learns to produce images that are convincing enough to "fool" the Discriminator.

## 🧠 Model Architectures

Both models are simple Multi-Layer Perceptrons (MLPs).

### Generator
The Generator takes a 100-dimensional latent vector (random noise) and upscales it to a 784-dimensional vector, which is then reshaped into a $28 \times 28$ image.

* **Input:** 100-dim latent vector ($z$)
* **Layer 1:** Linear (100 -> 256) + LeakyReLU
* **Layer 2:** Linear (256 -> 512) + LeakyReLU
* **Layer 3:** Linear (512 -> 1024) + LeakyReLU
* **Output Layer:** Linear (1024 -> 784) + **Tanh** (to scale output pixels between -1 and 1)

### Discriminator
The Discriminator takes a $28 \times 28$ image (flattened to 784 dimensions) and outputs a single probability of whether the image is real or fake.

* **Input:** 784-dim vector (flattened image)
* **Layer 1:** Linear (784 -> 1024) + LeakyReLU + Dropout
* **Layer 2:** Linear (1024 -> 512) + LeakyReLU + Dropout
* **Layer 3:** Linear (512 -> 256) + LeakyReLU + Dropout
* **Output Layer:** Linear (256 -> 1) + **Sigmoid** (to output a probability)

## 📊 Dataset: MNIST

The project uses the standard **MNIST dataset**, which consists of 60,000 training images and 10,000 testing images of $28 \times 28$ grayscale handwritten digits.

* Images are transformed into PyTorch Tensors.
* Pixel values are normalized to be in the `[-1, 1]` range (to match the Generator's `Tanh` output) using `transforms.Normalize((0.5,), (0.5,))`.

## 🛠️ Training Process

The models are trained for **100 epochs**. In each epoch, two steps occur:

1.  **Discriminator Training:**
    * The Discriminator is trained on a batch of **real images** (labeled as `1`).
    * The Generator creates a batch of **fake images** from random noise.
    * The Discriminator is trained on this batch of **fake images** (labeled as `0`).
    * The losses from both batches are combined, and the Discriminator's weights are updated.

2.  **Generator Training:**
    * The Generator creates a *new* batch of fake images.
    * These images are passed through the Discriminator.
    * The Generator's loss is calculated based on how well it "fooled" the Discriminator (i.e., how close the Discriminator's output was to `1`).
    * The Generator's weights are updated to get better at fooling the Discriminator.

* **Loss Function:** Binary Cross Entropy (`nn.BCELoss`)
* **Optimizer:** `Adam` (for both $G$ and $D$)
* **Checkpoints:** A grid of generated images and the Generator's model state are saved every 10 epochs.

## 🏎️ How to Run

1.  Ensure you have the required libraries installed:
    ```bash
    pip install torch torchvision matplotlib numpy
    ```
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open the `assignment3_problem2.ipynb` file.
4.  Run the cells sequentially from top to bottom. The notebook will:
    * Download the MNIST dataset.
    * Initialize the `Generator` and `Discriminator` models.
    * Begin the training loop for 100 epochs, printing the loss at each step.
    * Generate and save sample images every 10 epochs.
    * Load the final trained Generator and display a batch of newly generated digits.
