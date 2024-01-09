import numpy as np
from PIL import Image
import pandas as pd
import general_CNN_try1 as cnn
import sys
import general_MLP as mlp
import time
import scipy.signal


def time_func(function):
    def wrapper(*args):
        start = time.time()
        out = function(*args)
        end = time.time()
        print(end-start)
        return out
    return wrapper


def import_data(small=False, dev_perc=10):

    if small:
        data = pd.read_csv("Assets/small.csv")
    else:
        data = pd.read_csv("Assets/train.csv")
    data = np.array(data)
    m, n = data.shape

    num_dev = int(m*dev_perc / 100)

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


def display_example(original_img, conv_outp, example):

    show_pic(original_img[:, :, 0, example] * 255)
    for depth_idx in range(conv_outp.shape[2]):
        show_pic(conv_outp[:, :, depth_idx, example] * 255)
    pass


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

    # data fecth/normalize
    X_train, Y_train, X_dev, Y_dev = import_data(small=True, dev_perc=5)
    X_train, X_dev = mlp.normalize_data(
        X_train, X_dev, data_min=0, data_max=255)

    tot_examples = X_train.shape[1]
    batch_size = 50
    steps = 100
    batch_num = 1

    mlp_struct = (100, 10)

    layer1 = cnn.ConvLayer(input_shape=(28, 28, 1),
                           kernel_size=3,
                           output_depth=2)

    layer2 = cnn.ConvLayer(input_shape=(13, 13, 2),
                           kernel_size=3,
                           output_depth=4)
    
    w, b = mlp.init_params(mlp_struct, mode="X&G")

    for i in range(steps):
        batch_end = batch_size * batch_num
        if batch_end >= tot_examples:
            batch_num = 1
            batch_end = batch_num*batch_size
        else:
            batch_num += 1

        batch_start = batch_end - batch_size
        Y_batch = Y_train[batch_start:batch_end]

        X_batch = cnn.buildup(
            inp=X_train[:, batch_start:batch_end],
            target_shape=(28, 28, 1, batch_size))
        # X_batch: (height, width, channel, example) - normalized

        layer1.forward_prop(inp=X_batch,
                            activation_function="Leaky ReLU")
        layer1.pool(pool_size=2,
                    pool_type="mean")

        layer2.forward_prop(inp=layer1.p,
                            activation_function="Leaky ReLU")
        layer2.pool(pool_size=2,
                    pool_type="max")

        mlp_inp = layer2.p.reshape((100, batch_size), order="C")
        
        z, a = mlp.for_prop(inp=mlp_inp,
                            w=w,
                            b=b,
                            act_func="Leaky ReLU",
                            out_func="Softmax")

        dw, db, flat_dp2 = mlp.back_prop(inp=mlp_inp,
                                         z=z,
                                         a=a,
                                         w=w,
                                         Y=Y_batch,
                                         act_func="Leaky ReLU",
                                         out_func="Softmax",
                                         cost_func="CEL",
                                         dinp=True)

        
        dP2 = cnn.buildup(flat_dp2, (5, 5, 4, batch_size))

        layer2.unpool_actDerivative(dP2, "Leaky ReLU")
        layer2.backward_prop()

        layer1.unpool_actDerivative(layer2.dx, "Leaky ReLU")
        layer1.backward_prop()

        layer2.update_params(0.1)
        # layer1.update_params(0.1)

        w, b = mlp.update_params(w, b, dw, db, 0.01)
        
        print(mlp.get_accuracy(a[-1], Y_batch))

        


if __name__ == "__main__":
    main()
