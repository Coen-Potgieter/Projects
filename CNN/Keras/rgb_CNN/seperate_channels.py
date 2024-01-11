from tensorflow import keras
import numpy as np
import os
from PIL import Image


def load_data(file_path: str, import_data=False):
    '''
    Fetches/imports data from/to a .npz file

    Parameters:
    - file_path (str): Full path to the .npz file
    - import_data (bool): If the dataset isn't on the system, import_data=True 
        will load the data from keras and store it as a .npz file on the system

    Raises:
    ValueError: If file path is does not exist

    Returns:
    - X_train (3D array): (Examples, Height, Width)
    - Y_train (1D array): True Labels for each example
    - X_dev (3D array): (Examples, Height, Width)
    - Y_dev (1D array): True Labels for each example

    Note:
    - X_train, X_dev are not normalized (values = 0-255)
    '''

    def create_folder():
        '''
        Checks if file path exists, if it doesnt then creates it
        '''

        folder_path = file_path.split("/")
        folder_path.pop()
        folder_path = "/".join(folder_path)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    if import_data:
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        create_folder()
        np.savez(file_path, X_train=x_train,
                 Y_train=y_train, X_dev=x_test, Y_dev=y_test)
    else:
        if not os.path.exists(file_path):
            raise ValueError(f"File path '{file_path}' does not exist")

    npzfile = np.load(file_path)
    return npzfile["X_train"], npzfile["Y_train"], npzfile["X_dev"], npzfile["Y_dev"]


def show_pic(mat):
    '''
    Takes in a 3D array only when the last dimension is of 2,3 or 4
    so, 
        (h,w,1) gets rejected
        (h,w,2) works
        (h,w,3) works
        (h,w,4) works
        (h,w,5) gets rejected
    '''

    img = Image.fromarray(np.uint8(mat))
    scaled_img = img.resize((300, 300), Image.NEAREST)
    scaled_img.show()


def shuffle_data(X: np.ndarray, Y: np.ndarray, seed: int = None):
    '''
    Shuffles both X and Y keeping true labels attached to the correct X example

    Parameters:
    X (nD Array): X input to be shuffled (examples, ...)
    Y (nD Array): Y true labels to be shuffled (examples, ...)
    seed (int): Allows user to input a set seed for reproducibility 

    Explanation:
    - We set an initital 1d numpy array with values 0-Y.size this is the indicies
    - We then shuffle these indicies 
    - Then assing the shuffled indicies to both X and Y and return them

    Note:
    - Both X and Y must have Examples on the first axis, ie. (example, height, ...)
    '''
    if not seed is None:
        np.random.seed(seed)

    indices = np.arange(Y.size)
    np.random.shuffle(indices)

    return X[indices], Y[indices]


def one_hot(Y: np.ndarray, num_classes: int):
    '''
    One-hot encodes true labels

    Parameters:
    - Y (1D array): Each element the answer for each example
    - num_classes (int): Number of classes

    Returns:
    (2D array): (examples, true label layer)
    '''
    return np.eye(num_classes)[Y]

def sperate_channels(train, dev):

    num_channels = dev.shape[3]
    trains = []
    devs = []
    for channel in range(num_channels):
        trains.append(train[:,:,:,channel:channel+1])
        devs.append(dev[:,:,:,channel:channel+1])

    return trains, devs

def main():

    def compile_model(model):
        model.compile(optimizer="adam",
                      loss=keras.losses.CategoricalCrossentropy(
                          from_logits=False),
                      metrics=["accuracy"])

    def save_model(file_path):
        '''
        Saves the model in a .keras file

        Note:
        - can also seperatly save the weights using model.save_weights() function
            This is more lightweight and useful for transferring to already built,
            similar model architectures
        '''
        def create_folder():
            '''
            Checks if file path exists, if it doesnt then creates it
            '''
            folder_path = file_path.split("/")
            folder_path.pop()
            folder_path = "/".join(folder_path)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
        create_folder()
        model.save(filepath=file_path)
        return

    file_path = "Assets/Data/set.npz"
    X_train, Y_train, X_dev, Y_dev = load_data(
        file_path=file_path, import_data=True)
    X_train, Y_train = shuffle_data(X_train, Y_train)
    X_train = X_train / 255
    X_dev = X_dev / 255

    X_train_channels, X_dev_channels = sperate_channels(X_train, X_dev)
    Y_hot = one_hot(Y_train.flatten(), 10)

    inputs = []
    pooled2 = []
    for channel in range(3):
        inputs.append(keras.layers.Input(shape=(32, 32, 1)))
        conv = keras.layers.Conv2D(filters=3,
                                   kernel_size=3)(inputs[channel])
        activated = keras.layers.LeakyReLU(alpha=0.05)(conv)
        pooled = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        strides=2)(activated)
        conv2 = keras.layers.Conv2D(filters=3,
                                    kernel_size=3)(pooled)
        activated2 = keras.layers.LeakyReLU(alpha=0.05)(conv2)

        pooled2.append(keras.layers.MaxPool2D(pool_size=(2, 2),
                                              strides=2)(activated2))

    stacked = keras.layers.Concatenate()(pooled2)
    flattened = keras.layers.Flatten()(stacked)

    dense1 = keras.layers.Dense(150,
                                activation="tanh")(flattened)

    output = keras.layers.Dense(10,
                                activation="softmax")(dense1)
    model = keras.models.Model(inputs, output)
    compile_model(model)
    model.fit(x=X_train_channels, 
              y=Y_hot, 
              batch_size=100, epochs=5)

    file_path = "Assets/Models/Seperate-Channels.keras"
    save_model(file_path)


if __name__ == "__main__":
    main()
