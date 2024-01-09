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

    def build_model():
        '''
        Building the model

        Note:
        - data_format="channels_last" means (batch_size, height, width, channels)
            which makes more sense to me
        - should learn what 'adam' optimizer is, i think its velocity stuffs
        - the from_logits refers to if the Y will be one hotted or not, 
            if from_logits=False then it expects Y to be one-hotted
            
        '''
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(filters=2,
                                      kernel_size=3,
                                      strides=(1, 1),
                                      data_format="channels_last",
                                      use_bias=True))

        model.add(keras.layers.LeakyReLU(alpha=0.01))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2),
                                         strides=2,
                                         padding="valid",
                                         data_format="channels_last"))
        model.add(keras.layers.Flatten(data_format="channels_last"))
        model.add(keras.layers.Dense(model.layers[-1].output_shape[1],
                                     use_bias=True))

        model.add(keras.layers.Dense(10, activation="softmax",
                                     use_bias=True))

        model.compile(optimizer="adam",
                      loss=keras.losses.CategoricalCrossentropy(
                          from_logits=False),
                      metrics=["accuracy"])

    def learn():
        '''
        Performs learning
        '''
        model.fit(x=X_train, y=Y_hot_train.T, batch_size=100, epochs=8)

    def save_model():
        '''
        Saves the model in a .keras file

        Note:
        - can also seperatly save the weights using model.save_weights() function
            This is more lightweight and useful for transferring to already built,
            similar model architectures
        '''
        model.save(filepath="Assets/Models/model1.keras")

    def load_model():
        '''
        Loads model, must do more investigation on the architecture and type
            of this thing
        '''
        return keras.saving.load_model(filepath="Assets/Models/model1.keras")

    def predict():
        '''
        This is just forward prop and returning the last activation layer

        Note:
        - model.predict() computes the forward prop in batches becuase this is 
            more efficeint
        - mode.predict_on_batch() computes the forward prop for all the examples
            it is fed, it is meant for smaller size. 
        - If i were to feed both these functions the exact same (10_000, 784) input
            data, they would both spit out the same predictions, only .predict()
            would be much faster because, again, its working in batches
        '''

        # return model.predict_on_batch(X_dev)
        return model.predict(X_dev, batch_size=50)

    def evaluate():
        model.evaluate(x=X_dev, y=Y_hot_dev.T)

    file_path = "Assets/MNIST/digits.npz"
    X_train, Y_train, X_dev, Y_dev = load_data(file_path, import_data=False)
    X_train = X_train / 255
    X_dev = X_dev / 255
    Y_hot_dev = one_hot(Y_dev, 10)
    Y_hot_train = one_hot(Y_train, 10)

    model = load_model()
    # evaluate()
    predictions = predict()
    print(predictions.shape)
    print(predictions[0,:])
    # model.summary()

    # model = keras.Sequential()
    # build_model()
    # learn()
    # save_model()


if __name__ == "__main__":
    main()
