from tensorflow import keras
import pandas as pd
import numpy as np
import os 
from PIL import Image

def import_data(file_path, dev_split=10):
    '''
    Fetches input data from a .csv file

    Parameters:
    - file_path (str): File path where the .csv file is loacted
    - dev_split (int): Percentage of total examples be in Dev

    Raises:
    ValueError: If file path is does not exist

    Returns:
    - X_train (2D array): Pixel values for each example along columns (Nodes, Examples)
    - Y_train (1D array): True Labels for each example
    - X_dev (2D array): Pixel values for each example along columns (Nodes, Examples)
    - Y_dev (1D array): True Labels for each example

    Note:
    - X_train, X_dev are not normalized (values = 0-255)
    '''

    if not os.path.exists(file_path):
        raise ValueError(f"File path '{file_path}' does not exist")
    
    data = pd.read_csv(file_path)
    
    data = np.array(data)
    m, n = data.shape

    # split train/dev
    num_dev = int(m*dev_split / 100)

    # shuffle before splitting into dev and training sets
    np.random.shuffle(data)

    data_dev = data[:num_dev,].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:]
    X_dev = X_dev

    data_train = data[num_dev:,].T
    Y_train = data_train[0]
    X_train = data_train[1:]
    X_train = X_train


    return X_train, Y_train, X_dev, Y_dev

    # ok so, each column is each example, that means our 28x28 pixel image will have
    # 784 pixels in total so there are 784 rows in each of these matrices
    # The answers, our Y, is a 1 dimensionsal array, i think its a coloumn matrix
    # range of our X is 0 - 255

def one_hot(Y: np.ndarray, num_classes: int):
    '''
    One-hot encodes true labels

    Parameters:
    - Y (1D array): Each element the answer for each example
    - num_classes (int): Number of classes

    Returns:
    (2D array): (true label layer, examples)
    '''
    return np.eye(num_classes)[Y].T

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

def main():

    model = keras.saving.load_model("Assets/Models/model1.keras", compile=False)
    print(len(model.weights))

    return

    file_path = "../Assets/train.csv"
    X_train, Y_train, X_dev, Y_dev = import_data(file_path)

    
    X_train = np.reshape(X_train.T, (Y_train.size, 28,28)) / 255
    

    # show_pic(X_train[0,:,:])
    # return
    
    
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28,1)))
    model.add(keras.layers.Conv2D(filters=2,
                                  kernel_size=3,
                                  strides=(1,1),
                                  data_format="channels_last",
                                  use_bias=True))
    
    model.add(keras.layers.LeakyReLU(alpha=0.01))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2),
                                     strides=2,
                                     padding="valid",
                                     data_format="channels_last"))
    model.add(keras.layers.Flatten(data_format="channels_last"))
    model.add(keras.layers.Dense(model.layers[-1].output_shape[1],
                                 use_bias=True))

    model.add(keras.layers.Dense(10, activation="softmax",
                                 use_bias=True))
    
    
    model.compile(optimizer="adam",
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    
    Y_hot = one_hot(Y_train, 10)


    
    model.fit(X_train, Y_hot.T, batch_size=100, epochs=15)
    
    model.save("Assets/Models/model1.keras")
    model.save_weights("Assets/Params/model1.keras")

    pass

if __name__ == "__main__":
    main()
