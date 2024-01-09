from tensorflow import keras
import pandas as pd
import numpy as np
import os 
import sys

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
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        create_folder()
        np.savez(file_path, X_train=x_train, 
             Y_train=y_train, X_dev=x_test, Y_dev=y_test)
    else:
        if not os.path.exists(file_path):
            raise ValueError(f"File path '{file_path}' does not exist")
    
    npzfile = np.load(file_path)

    return npzfile["X_train"], npzfile["Y_train"], npzfile["X_dev"], npzfile["Y_dev"]
    

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


def main():
    
    X_train, Y_train, X_dev, Y_dev = load_data(file_path="Assets/MNIST/digits.npz", import_data=False)
    
    
    return
    
    
    X_train /= 255  

    model = keras.Sequential()
    model.add(keras.Input(shape=(784,)))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    model.add(keras.layers.Dense(135))
    model.add(keras.layers.LeakyReLU(alpha=0.01))

    model.add(keras.layers.Dense(10, activation="softmax"))

    
    model.compile(optimizer="adam",
                  loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    
    Y_hot = one_hot(Y_train, 10)
    print(Y_hot.shape)

    # keras takes in (num_examples, nodes)
    # so would access Xtrain example 1 as X_train(0,:)
    model.fit(X_train.T, Y_hot.T, batch_size=100, epochs=10)
    
    model.save
    pass

if __name__ == "__main__":
    main()
