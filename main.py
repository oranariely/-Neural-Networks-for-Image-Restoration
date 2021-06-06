import itertools
import json
import os
import random
import re
from glob import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from imageio import imread
from scipy.ndimage import zoom
from scipy.ndimage.filters import convolve
from skimage import color
from skimage.color import rgb2gray
from skimage.draw import line
from sklearn.metrics import classification_report
# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, UpSampling2D, Dense, Flatten, Reshape, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


########## Utils ##########

def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size - 1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def read_image(filename, representation):
    """Reads an image, and if needed makes sure it is in [0,1] and in float64.
    arguments:
    filename -- the filename to load the image from.
    representation -- if 1 convert to grayscale. If 2 keep as RGB.
    """
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im).astype(np.float64)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    return im


########## End of utils ##########

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A generator for generating pairs of image patches, corrupted and original
    :param filenames: a list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy array representation of an image as a single argument, and returning a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return:outputs random tuples of the form (source_batch, target_batch), where each output variable is an array of shape(batch_size, height, width, 1).
     target_batch is made of clean images and source_batch is their respective randomly corrupted version
     according to corruption_func(im)
    """
    names_image_dict = dict()
    h, w = crop_size
    while True:
        clean = np.zeros((batch_size, h, w))
        corrupt = np.zeros((batch_size, h, w))
        for i in range(batch_size):
            ind = np.random.choice(len(filenames))
            name = filenames[ind]
            if (name not in names_image_dict):
                names_image_dict[name] = read_image(name, 1)
            im = names_image_dict[name]
            x_random_min = np.random.choice(im.shape[1] - (w * 3) - 1)
            y_random_min = np.random.choice(im.shape[0] - (h * 3) - 1)
            x_random_max = x_random_min + (w * 3)
            y_random_max = y_random_min + (h * 3)
            patch_3x = im[y_random_min: y_random_max, x_random_min: x_random_max]
            cor_patch_3x = corruption_func(patch_3x)
            x_random_min = np.random.choice(patch_3x.shape[1] - w - 1)
            y_random_min = np.random.choice(patch_3x.shape[0] - h - 1)
            x_random_max = x_random_min + w
            y_random_max = y_random_min + h
            patch = patch_3x[y_random_min: y_random_max, x_random_min: x_random_max]
            cor_patch = cor_patch_3x[y_random_min: y_random_max, x_random_min: x_random_max]
            clean[i] = patch - 0.5
            corrupt[i] = cor_patch - 0.5
        clean = np.expand_dims(clean, 3)
        corrupt = np.expand_dims(corrupt, 3)
        yield corrupt, clean


def resblock(input_tensor, num_channels):
    """
    Takes as input a symbolic input tensor and the number of channels for each of its convolutional layers, and returns the symbolic output tensor of the resnet block.
    The convolutional layers should use “same” border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor: input tensor
    :param num_channels: number of channels
    :return: symbolic output tensor of the resnet block
    """
    conv1 = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    relu1 = Activation('relu')(conv1)
    conv2 = Conv2D(num_channels, (3, 3), padding='same')(relu1)
    add = Add()([input_tensor, conv2])
    output = Activation('relu')(add)
    return output


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Create an untrained Keras model with input dimension the shape of (height, width, 1), and all convolutional layers (including residual
    blocks) with number of output channels equal to num_channels, except the very last convolutional layer which should have a single output channel.
    The number of residual blocks should be equal to num_res_blocks.
    :param height: height
    :param width: width
    :param num_channels: number of channels
    :param num_res_blocks: number of residual blocks
    :return: an untrained Keras model.
    """
    input = Input(shape=(height, width, 1))
    conv1 = Conv2D(num_channels, (3, 3), padding='same')(input)
    relu1 = Activation('relu')(conv1)
    res_block = relu1
    for blocks in range(num_res_blocks):
        res_block = resblock(res_block, num_channels)
    conv2 = Conv2D(1, (3, 3), padding='same')(res_block)
    add = Add()([input, conv2])
    model = Model(inputs=input, outputs=add)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split, and generate from each set a dataset with the given batch size
    and corruption function. Eventually it will train the model.
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and should append anything to them.
    :param corruption_func: a corruption function.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: the number of update steps in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    np.random.shuffle(images)
    t_set = images[: np.int(len(images) * 0.8)]
    v_set = images[np.int(len(images) * 0.8):]
    t_generator = load_dataset(t_set, batch_size, corruption_func, (model.input_shape[1],
                                                                    model.input_shape[2]))
    v_generator = load_dataset(v_set, batch_size, corruption_func, (model.input_shape[1],
                                                                    model.input_shape[2]))
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(t_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=v_generator, use_multiprocessing=True,
                        validation_steps=(num_valid_samples // batch_size))


def restore_image(corrupted_image, base_model):
    """
    Restore full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of type float64 that is affected
    by a corruption generated from the same corruption function encountered during training (the image is not necessarily from the training set though).
    :param base_model: a neural network trained to restore small patches. The input and output of the network are images with values in the [−0.5, 0.5] range.
    :return: the restored image
    """
    corrupted_image -= 0.5
    a = Input((corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    im = new_model.predict(corrupted_image[np.newaxis, ..., np.newaxis])[0] + 0.5
    return np.clip(im.reshape(corrupted_image.shape), 0, 1).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random gaussian noise to an image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    """
    sig = np.random.uniform(min_sigma, max_sigma)
    corupt_im = image + np.random.normal(0, sig, image.shape)
    return np.around((corupt_im * 255) / 255).clip(0, 1)


num_res_blocks = 6  # @param {type:"slider", min:1, max:15, step:1}


def add_gaussian_noise_warper(image):
    return add_gaussian_noise(image, 0, 0.2)


def learn_denoising_model(num_res_blocks, quick_mode=False):
    """
    Train a denoising model
    :param num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    den_image = images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        train_model(model, den_image, add_gaussian_noise_warper, 10, 3, 2, 30)
    else:
        train_model(model, den_image, add_gaussian_noise_warper, 100, 100, 10, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: blurred image
    """
    return convolve(image, motion_blur_kernel(kernel_size, angle))


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Simulate motion blur on the given image using a square kernel of size kernel_size where the line has the given angle in radians, measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: blurred image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    corupt_im = add_motion_blur(image, kernel_size, angle)
    return np.around((corupt_im * 255) / 255).clip(0, 1)


# deblur_num_res_blocks = 6  # @param {type:"slider", min:1, max:15, step:1}


def random_motion_blur_warper(image):
    return random_motion_blur(image, [7])


def learn_deblurring_model(num_res_blocks, quick_mode=False):
    """
    Train a deblurring model
    :param num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    den_image = images_for_denoising()
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        train_model(model, den_image, random_motion_blur_warper, 10, 3, 2, 30)
    else:
        train_model(model, den_image, random_motion_blur_warper, 100, 100, 10, 1000)
    return model


def super_resolution_corruption(image):
    """
    Perform the super resolution corruption
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :return: corrupted image
    """
    zoom_num = np.random.choice([1 / 4, 1 / 3, 1 / 2])
    corupt_im = zoom(image, zoom_num)
    orig_size_im = zoom(corupt_im, [image.shape[0] / corupt_im.shape[0],
                                    image.shape[1] / corupt_im.shape[1]])
    return np.around((orig_size_im * 255) / 255).clip(0, 1)


# super_resolution_num_res_blocks = 3  # @param {type:"slider", min:1, max:15, step:1}
batch_size = 33  # @param {type:"slider", min:1, max:128, step:16}
steps_per_epoch = 1000  # @param {type:"slider", min:100, max:5000, step:100}
num_epochs = 8  # @param {type:"slider", min:1, max:20, step:1}
patch_size = 26  # @param {type:"slider", min:8, max:32, step:2}
num_channels = 42  # @param {type:"slider", min:16, max:64, step:2}


def super_resolution_corruption_warper(image):
    return super_resolution_corruption(image)


def learn_super_resolution_model(num_res_blocks, quick_mode=False):
    """
    Train a super resolution model
    :param num_res_blocks: number of residual blocks
    :param quick_mode: is quick mode
    :return: the trained model
    """
    den_image = images_for_denoising()
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        train_model(model, den_image, super_resolution_corruption_warper, 10, 3, 2, 30)
    else:
        train_model(model, den_image, super_resolution_corruption_warper, 100, 100, 10, 1000)
    return model

