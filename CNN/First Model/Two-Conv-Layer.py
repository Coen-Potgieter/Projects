import numpy as np
from PIL import Image
import pandas as pd
import general_CNN_fast as cnn
import general_MLP as mlp
import time
import sys


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

def visualize_feature_maps(X_batch, pooled_layer, example):
    show_pic(X_batch[:,:,0,example] * 255) 
    for depth_idx in range(pooled_layer.shape[2]):
        show_pic(pooled_layer[:,:,depth_idx,example] * 255)

def main():

    # data fecth/normalize
    X_train, Y_train, X_dev, Y_dev = import_data(small=False, dev_perc=5)
    X_train, X_dev = mlp.normalize_data(
        X_train, X_dev, data_min=0, data_max=255)

    tot_examples = Y_train.size
    batch_size = 50
    steps = 1000
    batch_num = 1

    mlp_struct = (100, 10)

    layer1 = cnn.ConvLayer(input_shape=(28, 28, 1),
                           kernel_size=3,
                           output_depth=2)
    
    layer2 = cnn.ConvLayer(input_shape=(13, 13, 2),
                           kernel_size=3,
                           output_depth=4)
    
    # load params and assign
    # k_list, b_list = cnn.load_params(file_path="Assets/Params/CNN")
    # layer1.k, layer1.b = k_list[0], b_list[0]
    # layer2.k, layer2.b = k_list[1], b_list[1]
    

    w, b = mlp.init_params(mlp_struct, mode="X&G")
    # w,b = mlp.load_params(file_path="Assets/Params/MLP")

    cnn_lr = 0.01
    for step in range(steps):
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

        layer1.forward_prop(inp=X_batch,
                            activation_function="Leaky ReLU")
        layer1.pool(pool_size=2,
                    pool_type="max")
        
        layer2.forward_prop(inp=layer1.p,
                            activation_function="ReLU")
        layer2.pool(pool_size=2,
                    pool_type="max")
        
        # visualize_feature_maps(X_batch, layer1.p, 12)
        # sys.exit()
    
        mlp_inp = layer2.p.reshape((mlp_struct[0], batch_size), order="C")

        z, a = mlp.for_prop(inp=mlp_inp,
                            w=w,
                            b=b,
                            act_func="Leaky ReLU",
                            out_func="Tanh")

        dw, db, flat_dp = mlp.back_prop(inp=mlp_inp,
                                         z=z,
                                         a=a,
                                         w=w,
                                         Y=Y_batch,
                                         act_func="Leaky ReLU",
                                         out_func="Tanh",
                                         cost_func="MSE",
                                         dinp=True)

        dp = cnn.buildup(flat_dp, layer2.p.shape)

        layer2.unpool_actDerivative(dp)
        layer2.backward_prop()

        layer1.unpool_actDerivative(layer2.dx)
        layer1.backward_prop()

        layer1.dk = np.clip(layer1.dk, -1, 1)
        layer2.dk = np.clip(layer2.dk, -1, 1)

        layer1.update_params(cnn_lr)
        layer2.update_params(cnn_lr)

        w, b = mlp.update_params(w, b, dw, db, 0.1)

        if step % 10 == 0:
            # print(layer2.p[:,:,1,0])
            print(f"Iteration {step}")
            print(mlp.get_accuracy(a[-1], Y_batch))

        if np.any(np.isnan(w[-1])):
            # incredibly crude solution, sorry
            print(mlp_inp[:,0])
            print("\n\nNaNed bro")
            w, b = mlp.init_params(mlp_struct, mode="X&G")
            layer1.k, layer1.b = cnn.init_params((28, 28, 1),3,2)


    mlp.save_params(w, b, "Assets/Params/MLP")
    cnn.save_params(kernels=[layer1.k, layer2.k],
                    biases=[layer1.b, layer2.b],
                    file_path="Assets/Params/CNN")


if __name__ == "__main__":
    main()
