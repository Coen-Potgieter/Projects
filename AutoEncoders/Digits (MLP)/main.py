import tensorflow as tf
print(tf.__version__)
print(tf.keras.__file__)
import os
import numpy as np
import mlp_autoEncoder
from matplotlib import pyplot as plt
import pygame_UI


def load_data(file_path: str, import_data=False):
    '''
    Fetches/imports data from/to a .npy files

    Parameters:
    - file_path (str): path to the folder where the .npy files are stored
    - import_data (bool): If the dataset isn't on the system, import_data=True 
        will load the data from keras and store it as a .npy files on the system

    Raises:
    ValueError: If file path is does not exist

    Returns:
    - X_train (4D array): (60k, 28, 28, 1)
    - X_dev (4D array): (10k, 28, 28, 1)

    Note:
    - X_train, X_dev are normalized (values range from 0-1)
    - saves values as float16 to save space
    '''

    def create_folder():
        '''
        Checks if file path exists, if it doesnt then creates it
        '''

        if not os.path.exists(file_path):
            os.makedirs(file_path)

    if import_data:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        X_train = np.expand_dims((x_train / 255).astype(np.float16), axis=-1)
        X_dev = np.expand_dims((x_test / 255).astype(np.float16), axis=-1)

        create_folder()
        np.save(file_path + "/X_train.npy", X_train)
        np.save(file_path + "/X_dev.npy", X_dev)

    else:
        if not os.path.exists(file_path):
            raise ValueError(f"File path '{file_path}' does not exist")

    X_train = np.load(file_path + "/X_train.npy")
    X_dev = np.load(file_path + "/X_dev.npy")

    assert X_train.shape == (60_000, 28, 28, 1)
    assert X_dev.shape == (10_000, 28, 28, 1)
    assert np.max(X_train) == 1.0

    return X_train,  X_dev


def extract_decoder(model):
    decoder_inp = model.get_layer("d1").input
    decoder_outp = model.get_layer("Output").output
    return tf.keras.models.Model(decoder_inp, decoder_outp, name="decoder")


def extract_encoder(model):
    ecoder_inp = model.get_layer("Flatten").input
    ecoder_outp = model.get_layer("Latent").output
    return tf.keras.models.Model(ecoder_inp, ecoder_outp, name="encoder")


def test_auto(auto_path, X):
    '''
    Parameters:
    - auto_path (str): file path to the autoencoder model
    - X (4D array): (Examples, 28, 28, 1)
    '''
    auto_encoder = tf.keras.saving.load_model(auto_path)
    reconstruction = auto_encoder.predict(X, verbose=0)
    plot_imgs((X, reconstruction), ("Original", "Reconstruction"))


def plot_imgs(data: tuple, titles: tuple):
    '''
    Produces grid of images from given numpy arrays

    Parameters:
    - data (tuple): List of 4D arrays (batch size, height, width, channel)
        Each array being an image to be displayed
    - titles (tuple): List of titles associated with each batch of arrays

    Raises:
    - ValueError: If number of batches don't match up with number of titles given

    Note:
    - Every batch given opens its own window
    '''
    if len(data) != len(titles):
        raise ValueError

    plt.style.use("dark_background")
    for idx in range(len(data)):
        num_examples = data[idx].shape[0]
        grid_size = int(np.ceil(np.sqrt(num_examples)))
        fig, sub_axes = plt.subplots(grid_size, grid_size,
                                     sharex=True, sharey=True,
                                     figsize=(7, 7))
        fig.suptitle(titles[idx],
                     size="xx-large",
                     weight="bold",
                     y=0.95)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        for ax in sub_axes.ravel():
            ax.set_axis_off()

        example_idx, inner_break = 0, False
        for y in range(grid_size):
            if inner_break:
                break
            for x in range(grid_size):
                try:
                    sub_axes[y, x].imshow(
                        data[idx][example_idx], cmap='gray')
                except IndexError:
                    inner_break = True
                    break
                example_idx += 1
    plt.show()


def generate_digits(path):
    '''
    Runs pygame UI
    '''

    auto = tf.keras.models.load_model(path, compile=False)
    decoder = extract_decoder(auto)
    pygame_UI.run(decoder)


def latent_space_inference(path, X):
    '''
    Gains insight into the properties of the latent vectors

    Parameters:
    - path (str): file path to the model
    - X (4D array): (Examples, 28, 28, 1)

    Raises:
    - ValueError: If file path does not exist

    Note:
    - Not only is a graph that displays the distributions of each node produced
        but also a printed table in terminal output

    Evaluation:
    Can see distribution is not normal, mean is around 7.5-8, with std 
    deviation being around 4. Also is interesting to see how nodes 3,4 aren't
    activated in any way. I suppose then to get nice results for image generation 
    keep sliders 3 and 4 at zero and make sure the values are less than 15 or so.

    Note:
    - These evaluations vary alot if were to train the system again. 
        (As in completely different)
    '''

    if not os.path.exists(path):
        raise ValueError(f"File path '{path}' does not exist")

    auto = tf.keras.saving.load_model(path)
    encoder = extract_encoder(auto)
    latent = encoder.predict(X, verbose=0)

    lo, hi, mean, sd = np.min(latent, axis=0), np.max(latent, axis=0), \
        np.mean(latent, axis=0), np.std(latent, axis=0)

    num_data_pts, num_nodes = latent.shape
    bin_num = np.ceil(np.sqrt(num_data_pts)).astype(np.uint8)
    grid_size = int(np.ceil(np.sqrt(num_nodes)))

    plt.style.use("Solarize_Light2")
    fig, sub_axes = plt.subplots(grid_size, grid_size,
                                 sharex=True, sharey=False,
                                 figsize=(7, 7))
    fig.suptitle("Distribution of values for each node",
                 size="xx-large",
                 weight="bold",
                 y=0.95)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85,
                        hspace=0.4)

    lowest_lo, highest_hi = np.min(lo), np.max(hi)
    node_idx, inner_break = 0, False
    for y in range(grid_size):
        if inner_break:
            break
        for x in range(grid_size):
            try:
                sub_axes[y, x].hist(latent[:, node_idx], bins=bin_num)
                sub_axes[y, x].set(xlim=(lowest_lo, highest_hi))
                sub_axes[y, x].set_title(f"Node {node_idx+1}",
                                         size="medium")
            except IndexError:
                inner_break = True
                break
            node_idx += 1

    for ax in sub_axes.ravel()[node_idx:]:
        ax.set_axis_off()

    print("|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|".format(
        "Node", "Min", "Max", "Mean", "std"))
    print("----------------------------------------------")
    for idx_node in range(lo.shape[0]):
        print("|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|".format(
            str(idx_node+1),
            str(np.round(lo[idx_node], 2)),
            str(np.round(hi[idx_node], 2)),
            str(np.round(mean[idx_node], 2)),
            str(np.round(sd[idx_node], 2))))
    plt.show()


def learn(X_train, auto_path, init=True):
    '''
    Trains autoencoder

    Parameters:
    - X_train (4D array): Data to be trained on (Examples, height, width, channels)
    - auto_path (str): File path to the auto-encoder model 
    - init (bool): initialises new weights and biases if set to True

    Specifications:
    - MSE
    '''

    if init:
        auto_encoder = mlp_autoEncoder.build(X_train.shape[1:])
    else:
        auto_encoder = tf.keras.saving.load_model(auto_path)

    auto_encoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                         loss=tf.keras.losses.MeanSquaredError(),
                         metrics=["accuracy"])

    auto_encoder.fit(X_train, X_train, batch_size=32, epochs=10)

    auto_encoder.save(auto_path, save_format="keras")


def main():
    auto_path = "Assets/Models/auto"

    X_train, X_dev,  = load_data("Assets/Data/Numpy", import_data=True)

    generate_digits(auto_path)

    # test_auto(auto_path, X_dev[50:75])

    # latent_space_inference(auto_path, X_dev)

    # learn(X_train, auto_path, init=True)


if __name__ == "__main__":
    main()
