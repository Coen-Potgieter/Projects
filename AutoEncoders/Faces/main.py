import numpy as np
import tensorflow as tf
import os
import cnn_autoEncoder
from matplotlib import pyplot as plt
import pygame_gen_faces
import pygame_draw_faces


def load_arrays(npy_path: str):
    '''
    Loads in numpy arrays

    Parameters:
    - npy_path (str): Path to the folder where all npy files are stored

    Rasies:
    - ValueError: File path does not exist

    Returns:
    - X_train (4D array): Grayscale (examples, height, width, 1)
    - X_dev (4D array): Grayscale (examples, height, width, 1)

    Note:
    - Output arrays are normalized and of type float16 
        (ie. values range from 0-1)

    '''
    if not os.path.exists(npy_path):
        raise ValueError(f"File path '{npy_path}' does not exist")

    X_train, X_dev = np.load(
        npy_path + "/X_train.npy"), np.load(npy_path + "/X_dev.npy")

    assert X_train.shape == (1_000, 171, 186, 1)
    assert X_dev.shape == (500, 171, 186, 1)

    return X_train, X_dev


def learn(X_train, X_dev, auto_path, init=True):
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
        auto_encoder = cnn_autoEncoder.build(X_train.shape[1:])
    else:
        auto_encoder = tf.keras.saving.load_model(auto_path)

    _, outp_h, outp_w, _ = auto_encoder.layers[-1].output.shape

    start_h = (X_train.shape[1] - outp_h) // 2
    end_h = start_h + outp_h
    start_w = (X_train.shape[2] - outp_w) // 2
    end_w = start_w + outp_w

    auto_encoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                         loss=tf.keras.losses.MeanSquaredError())

    auto_encoder.fit(
        x=X_train, y=X_train[:, start_h:end_h, start_w:end_w, :],
        batch_size=8,
        epochs=5,
        validation_data=(X_dev, X_dev[:, start_h:end_h, start_w:end_w, :]),
    )

    auto_encoder.save(auto_path)


def test_auto(auto_path, X):
    '''
    Parameters:
    - auto_path (str): file path to the autoencoder model
    - X (4D array): (Examples, height, width, channels)
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


def extract_decoder(model):
    decoder_inp = model.get_layer("Decoder-Start").input
    decoder_outp = model.get_layer("Decoder-End").output
    return tf.keras.models.Model(decoder_inp, decoder_outp, name="decoder")


def extract_encoder(model):
    ecoder_inp = model.get_layer("Encoder-Start").input
    ecoder_outp = model.get_layer("Encoder-End").output
    return tf.keras.models.Model(ecoder_inp, ecoder_outp, name="encoder")


def latent_space_inference(path, X, num_plots=None):
    '''
    Gains insight into the properties of the latent vectors

    Parameters:
    - path (str): file path to the model
    - X (4D array): (Examples, 28, 28, 1)
    - num_plots (int): Determines the Number of node distributions to be shown,
        Will show all non-zero nodes if not specified

    Raises:
    - ValueError: If file path does not exist

    Note:
    - Not only is a graph that displays the distributions of each node produced
        but also a printed table in terminal output

    Evaluation:
    - Distributions behave somewhat nicely, with a mean of around 2 and deviation of 1.
    - Observing 28 non activated nodes might be concerning but I'm not sure

    Note:
    - These evaluations vary alot if were to train the system again. 
        (As in completely different)
    - For image generation I am going to use total activation of a node across all examples
        as the measure of influence in the output. Will see if this is a good idea or 
        not later I suppose. 
    '''

    if not os.path.exists(path):
        raise ValueError(f"File path '{path}' does not exist")

    auto = tf.keras.saving.load_model(path)
    encoder = extract_encoder(auto)
    latent = encoder.predict(X, verbose=0)

    total_activation = np.sum(latent, axis=0)
    non_zero_nodes = np.where(total_activation != 0)[0]

    lo, hi, mean, sd = np.min(latent, axis=0), np.max(latent, axis=0), \
        np.mean(latent, axis=0), np.std(latent, axis=0)

    if num_plots is None:
        num_plots = non_zero_nodes.size
    elif num_plots > non_zero_nodes.size:
        raise ValueError(
            f"Number of plots given '{num_plots}' is greater than the number of non-zero output nodes: {non_zero_nodes.size}")

    # method grabbed from internet to get bin size
    bin_num = np.ceil(np.sqrt(latent.shape[0])).astype(np.uint8)
    grid_size = int(np.ceil(np.sqrt(num_plots)))

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
        if inner_break:  # This method exists both loops
            break
        for x in range(grid_size):
            # if we index something out of range then there is no more nodes to plot
            try:
                # we are using the non-zero-nodes as our index
                sub_axes[y, x].hist(
                    latent[:, non_zero_nodes[node_idx]], bins=bin_num)
                sub_axes[y, x].set(xlim=(lowest_lo, highest_hi))
                sub_axes[y, x].set_title(f"Node {non_zero_nodes[node_idx]+1}",
                                         size="medium")
            except IndexError:
                inner_break = True
                break
            node_idx += 1

    # turn of axis where nothing is being plotted
    for ax in sub_axes.ravel()[node_idx:]:
        ax.set_axis_off()

    print("|{:^8}|{:^8}|{:^8}|{:^8}|{:^12}|".format(
        "Node", "Max", "Mean", "dev", "Total Acts"))
    print("--------------------------------------------------")
    for idx_node in range(latent.shape[1]):
        print("|{:^8}|{:^8}|{:^8}|{:^8}|{:^12}|".format(
            str(idx_node+1),
            str(np.round(hi[idx_node], 2)),
            str(np.round(mean[idx_node], 2)),
            str(np.round(sd[idx_node], 2)),
            str(np.round(total_activation[idx_node], 2))))

    avg_mean = np.mean(mean[non_zero_nodes])
    avg_sd = np.round(np.mean(sd[non_zero_nodes]), decimals=2)

    # Assures me that deviation is not big
    assert np.std(mean[non_zero_nodes]) < 1
    assert np.std(sd[non_zero_nodes]) < 1

    num_zero_nodes = latent.shape[1] - non_zero_nodes.size
    print(f"\nNumber of zero nodes = {num_zero_nodes}")
    print("Mean across all non-zero nodes = {:.2f}".format(avg_mean))
    print("Deviation across all non-zero nodes = {:.2f}".format(avg_sd))
    print("Maximum value across all nodes = {:.2f}".format(highest_hi))
    print("Finally, the deviation of both mean and devation across all non-zero nodes is less than 1\n")

    plt.show()


def generate_faces(path, X):
    '''
    Runs pygame UI
    '''
    auto = tf.keras.saving.load_model(path)
    decoder = extract_decoder(auto)
    encoder = extract_encoder(auto)
    latent = encoder.predict(X, verbose=0)

    pygame_gen_faces.run(decoder, latent_vectors=latent)


def draw_faces(path):
    auto = tf.keras.saving.load_model(path)
    pygame_draw_faces.main(auto)
    pass


def main():

    auto_path = "Assets/Models/auto.keras"
    X_train, X_dev = load_arrays("Assets/Data/Numpy")

    # plot_imgs((X_train[:100],), ("faces",))

    # learn(X_train[15_000:, :, :, :],
    #       X_dev[3000:4000, :, :, :], auto_path, init=False)

    # test_auto(auto_path, X_dev[50:75])

    # latent_space_inference(auto_path, X_dev, num_plots=25)

    generate_faces(auto_path, X_dev[:500])

    # draw_faces(auto_path)


if __name__ == "__main__":
    main()
