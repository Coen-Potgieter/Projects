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
    
    show_pic(original_img[:,:,0,example] * 255)
    for depth_idx in range(conv_outp.shape[2]):
        show_pic(conv_outp[:,:,depth_idx, example]* 255)
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

    # conv1 = {
    #     "kernel size": 3,
    #     "number of kernels": 2
    # }
    # pool1 = {
    #     "size": 2,
    #     "type": "mean",
    #     "activation": "Leaky ReLU"
    # }
    # conv2 = {
    #     "kernel size": 3,
    #     "number of kernels": 4
    # }
    # pool2 = {
    #     "size": 2,
    #     "type": "max",
    #     "activation": "Leaky ReLU"
    # }

    def single_example_for(inp, example):

        X1 = inp[:,:,:,example]

        # conv1
        k1, b1 = cnn.init_params(input_shape=X1.shape,
                                 kernel_size=3,
                                 output_depth=2)
        
        z1 = cnn.forward(inp=X1,
                           k=k1,
                           b=b1)
        
        c1 = cnn.apply_activation(z1, "Leaky ReLU")

        # pool1
        p1, map_p1 = cnn.pool(c1, 2, "Max")
        
        print(p1.shape)
        sys.exit()
        # conv2
        k2, b2 = cnn.init_params(input_shape=p1.shape,
                                 kernel_size=3,
                                 output_depth=4)
        
        z2 = cnn.forward(p1, k2, b2)

        c2 = cnn.apply_activation(z2, "Leaky ReLU")

        # pool2
        p2, map_p2 = cnn.pool(c2, 2, "mean")

        K1[:,:,:,:,example] = k1
        B1[:,:,:,example] = b1
        Z1[:,:,:,example] = z1
        C1[:,:,:,example] = c1
        P1[:,:,:,example] = p1
        map_P1[:,:,:,example] = map_p1
        K2[:,:,:,:,example] = k2
        B2[:,:,:,example] = b2
        Z2[:,:,:,example] = z2
        C2[:,:,:,example] = c2
        P2[:,:,:,example] = p2
        map_P2[:,:,:,example] = map_p2

    def single_example_back(inp, example):
        pass

    # data fecth/normalize
    X_train, Y_train, X_dev, Y_dev = import_data(small=True, dev_perc=5)
    X_train, X_dev = mlp.normalize_data(
        X_train, X_dev, data_min=0, data_max=255)

    tot_examples = X_train.shape[1]
    batch_size = 100
    steps = 100
    batch_num = 1

    mlp_struct = (100, 10)



    for i in range(steps):

        # i know this is vile, i think i need to use a class or something
        K1 = np.zeros((3,3,1,2, batch_size))
        B1 = np.zeros((26,26,2, batch_size))
        Z1 = np.zeros((26,26,2, batch_size))
        C1 = np.zeros((26,26,2, batch_size))
        P1 = np.zeros((13,13,2, batch_size))
        map_P1 = np.zeros((26,26,2, batch_size))
        K2 = np.zeros((3,3,2,4, batch_size))
        B2 = np.zeros((11,11,4, batch_size))
        Z2 = np.zeros((11,11,4, batch_size))
        C2 = np.zeros((11,11,4, batch_size))
        P2 = np.zeros((5, 5, 4, batch_size))
        map_P2 = np.zeros((11,11,4, batch_size))

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

        # batch forward
        for example in range(X_batch.shape[3]):
            single_example_for(X_batch, example)
        

        mlp_inp = P2.reshape((100, batch_size), order="C")

        w, b = mlp.init_params(mlp_struct, mode="X&G")
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
        
        dP2 = cnn.buildup(inp=flat_dp2,  
                          target_shape=(5, 5, 4, 100))
        

        for example in range(dP2.shape[3]):
            single_example_back()
            




        sys.exit()
        


        w, b = mlp.update_params(w, b, dw, db, 0.1)

        # print(end-start)


if __name__ == "__main__":
    main()
