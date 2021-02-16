About
Corresponding code to the paper "Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini and David Wagner, at IEEE Symposium on Security & Privacy, 2017.

Implementations of the three attack algorithms in Tensorflow. It runs correctly on Python 3 (and probably Python 2 without many changes).

To evaluate the robustness of a neural network, create a model class with a predict method that will run the prediction network without softmax. The model should have variables

model.image_size: size of the image (e.g., 28 for MNIST, 32 for CIFAR)
model.num_channels: 1 for greyscale, 3 for color images
model.num_labels: total number of valid labels (e.g., 10 for MNIST/CIFAR)
