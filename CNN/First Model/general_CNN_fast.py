'''
This is not the final product, it cant handle multiple examples and batches
without introducing nested for loops but whatever, im just gonna do it
'''
import numpy as np
import os
from scipy.signal import correlate2d


class ConvLayer:
    def __init__(self, input_shape, kernel_size, output_depth):
        self.k, self.b = init_params(input_shape=input_shape,
                                     kernel_size=kernel_size,
                                     output_depth=output_depth)

    def forward_prop(self, inp, activation_function: str = None):

        self.act_func = activation_function
        self.x = inp
        self.num_examples = inp.shape[3]
        z_shape = (self.b.shape[0], self.b.shape[1],
                   self.b.shape[2], self.num_examples)
        self.z = np.zeros(z_shape)

        for example in range(self.num_examples):
            self.z[:, :, :, example] = _forward(inp=self.x[:, :, :, example],
                                                k=self.k,
                                                b=self.b)

        if not self.act_func is None:
            self.c = apply_activation(inp=self.z,
                                      act_func=self.act_func)
        else:
            self.c = self.z

    def pool(self, pool_size, pool_type, stride=None):

        self.pool_size = pool_size

        if stride is None:
            stride = pool_size
        p_shape = (
            (self.c.shape[0]-pool_size) // stride + 1,
            (self.c.shape[1]-pool_size) // stride + 1,
            (self.c.shape[2]),
            self.c.shape[3]
        )

        self.p = np.zeros(p_shape)
        self.pool_map = np.zeros(self.c.shape)
        for example in range(self.num_examples):
            self.p[:, :, :, example], self.pool_map[:, :, :, example] = _pool(inp=self.c[:, :, :, example],
                                                                              pool_size=pool_size,
                                                                              pool_type=pool_type)

    def unpool_actDerivative(self, dp2):

        self.dc = np.zeros(self.pool_map.shape)
        for example in range(self.num_examples):
            self.dc[:, :, :, example] = _unpool(inp=dp2[:, :, :, example],
                                                pool_map=self.pool_map[:,
                                                                       :, :, example],
                                                pool_size=self.pool_size)

        if not self.act_func is None:
            preact_z = apply_activation(inp=self.z,
                                        act_func=self.act_func,
                                        derivative=True)
            self.dz = self.dc * preact_z
        else:
            self.dz = self.dc

    def backward_prop(self):

        self.dk = np.zeros(self.k.shape)
        self.db = np.zeros(self.b.shape)
        self.dx = np.zeros(self.x.shape)

        for example in range(self.num_examples):
            dk, db, dx = _backward(self.dz[:, :, :, example],
                                   self.x[:, :, :, example],
                                   self.k)
            self.db += db
            self.dk += dk
            self.dx[:, :, :, example] = dx

        self.dk = self.dk / self.num_examples
        self.db = self.db / self.num_examples

    def update_params(self, lr):
        self.k = self.k - lr*self.dk
        self.b = self.b - lr*self.db


def init_params(input_shape: tuple, kernel_size: int, output_depth: int):
    '''
    Initializes Kernel and Biase tensors for a single 
        convolutional layer

    Parameters:
    - input_shape (tuple): Specifies the size of the input (height, width, input depth)
    - kernel_size (int): Size of the sqaure kernels to be used
    - output_depth (int): Number of filters to be used

    Returns:
    - output_shape (tuple): Output shape of the convolution layer (height, width, output depth)
    - k (4D Tensor): Kernel Tensor (height, width, input depth, output depth)
    - b (3D Tensor): bias for each filter (height, width, output depth)

    Note: 
    - Using `channels-last` convention, ie. (Height, Width, Input Depth, Output Depth)
    - The kernel size is used rather than the kernel radius
    - Output Depth is the number of filters in the layer
    - Input Depth is the number of kernels for each filter
    '''
    output_shape = [input_shape[0] - kernel_size + 1,
                    input_shape[1] - kernel_size + 1,
                    output_depth]

    kernel_shape = (kernel_size,
                    kernel_size,
                    input_shape[2],
                    output_depth)

    # k = np.random.randn(*kernel_shape)
    k = np.random.uniform(low=-0.2, high=0.2, size=kernel_shape)
    b = np.zeros(output_shape)

    return k, b


def _forward(inp, k, b):
    '''
    Performs the forward pass for a convolutional layer

    Parameters:
    - inp (np.ndarray): Input data with shape (height, width, input_depth)
    - k (4D Tensor): Kernel Tensor (height, width, input depth, output depth)
    - b (3D Tensor): bias for each filter (height, width, output depth)

    Notes:
    - The input data is expected to be in grayscale format
    - The convolutions for each channel are summed up into one matrix

    Returns:
    - out (3D Tensor): Result of the forward pass (height, width, output depth)
    '''
    output_shape = b.shape
    out = np.zeros(output_shape)
    for depth_idx in range(output_shape[2]):
        for channel in range(inp.shape[2]):

            single_k = k[:, :, channel, depth_idx]
            out[:, :, depth_idx] += correlate2d(
                inp[:, :, channel], single_k, mode="valid")

        out[:, :, depth_idx] += b[:, :, depth_idx]

    return out


def _pool(inp: np.ndarray, pool_size: int, pool_type: str, stride=None):
    """
    Perform pooling on a 2D Tensor.

    Parameters:
    - inp (3D Tensor): Input to be downsized (height, width, depth)
    - pool_size (int): Size of the sqaure pooling window
    - pool_type (str): Pooling type, ("max", "mean", "min")
    - stride (int): Stride. If None, it defaults to pool_size.

    Returns:
    - out (3D array): (height, width, depth)
    - pool_map (3D array): This will be a binary tensor that indicates wher 
        the max/min values were take from, it will be the same shape as out input
        (height, width, depth)
    """

    def apply_pooling(inp):
        return valid_pool_types[pool_type](inp)

    def map_pool(inp):
        '''
        Gets y, x coords for the max/min value in the extracted input
        '''

        flat_index = pool_map_types[pool_type](inp)
        return np.unravel_index(flat_index, (pool_size, pool_size))

    if stride is None:
        stride = pool_size

    valid_pool_types = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean
    }

    pool_map_types = {
        "max": np.argmax,
        "min": np.argmin
    }
    pool_type = pool_type.lower()

    if pool_type not in valid_pool_types.keys():
        raise ValueError(
            f"Invalid pool_type='{pool_type}', Valid Pool Types are {list(valid_pool_types.keys())}")

    out_shape = (
        (inp.shape[0]-pool_size) // stride + 1,
        (inp.shape[1]-pool_size) // stride + 1,
        inp.shape[2]
    )

    out = np.zeros(out_shape)
    pool_map = np.zeros(inp.shape)
    for depth_idx in range(inp.shape[2]):
        for y in range(out_shape[0]):
            y_start = y*stride
            y_end = y_start + pool_size
            for x in range(out_shape[1]):

                x_start = x*stride
                x_end = x_start + pool_size
                extracted_inp = inp[y_start:y_end,
                                    x_start:x_end,
                                    depth_idx]

                out[y, x, depth_idx] = apply_pooling(extracted_inp)

                if pool_type != "mean":

                    y_index, x_index = map_pool(extracted_inp)
                    pool_map[y_start+y_index,
                             x_start+x_index, depth_idx] = 1

    if pool_type == "mean":
        return out, np.ones((inp.shape)) * 1/(pool_size**2)
    else:
        return out, pool_map


def apply_activation(inp: np.ndarray, act_func: str, derivative=False):
    '''
    Applies a speicified activation function to all elements of 
    an nd array input

    Parameters:
    - inp (ND Array): Function will be applied to every element in this array
    - act_func (str): Specifies the activation function to be used
    - derivative (bool): Specifies whether to perform the derivative if the function or not

    Returns:
    - (ND Array): same dimensions as `inp`

    Raises:
    - ValueError: If an invalid activation function is given

    Note:
    - Valid act_func inputs are as follows:
        ["Leaky ReLU", "ReLU", "Tanh", "Sigmoid", "SiLU"]
    '''

    def ReLU(x):
        if derivative:
            return x > 0
        return np.maximum(x, 0)

    def hyp_tan(x):
        if derivative:
            return 1 - np.tanh(x) * np.tanh(x)
        return np.tanh(x)

    def leaky_ReLU(x, alpha=0.01):
        if derivative:
            return np.where(x >= 0, 1, alpha)
        return np.maximum(alpha * x, x)

    def sigmoid(x):
        if derivative:
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        return (1 / (1 + np.exp(-x)))

    def SiLU(x):
        if derivative:
            return (np.exp(x) + 1 + x * np.exp(x)) / (1 + np.exp(x))**2
        return x * sigmoid(x)

    valid_funcs = {
        "Leaky ReLU": leaky_ReLU,
        "ReLU": ReLU,
        "Tanh": hyp_tan,
        "Sigmoid": sigmoid,
        "SiLU": SiLU
    }

    if act_func not in valid_funcs.keys():
        raise ValueError(
            f"Invalid Activation Function `{act_func}`, Valid Functions are{list(valid_funcs.keys())}")

    return valid_funcs[act_func](inp)


def flatten(inp: np.ndarray):
    '''
    Flattens a 4D Tensor

    Parameters:
    - inp (4D Tensor): Input to be flattened (height, width, depth, example)

    Returns:
    - out (2D array): Flattens by depth, then row then column

    Note:
    - Order is arbitrairy if using the same order='C' parameter when builing up again
    - C-style (row-major order): Elements are stored row-wise. The last axis changes fastest.
    '''

    height, width, depth, batch_size = inp.shape
    return inp.reshape((height * width * depth, batch_size), order="C")


def _unpool(inp: np.ndarray, pool_map: np.ndarray, pool_size: int, stride=None):
    '''
    Unpools a pooled input

    Parameters:
    - inp (3D array): Lower dimensioned pooled tensor (height, width, depth)
    - pool_map (3D array): Higher dimensioned binary tensor (height, width, depth)
    - pool_size (int): Size of the sqaure pooling window
    - stride (int): Stride. If None, it defaults to pool_size.

    Returns:
    - pool_map (3D array): all the ones are replace by their respecitve values
    '''

    if stride is None:
        stride = pool_size

    inp_height, inp_width, num_depth = inp.shape

    for depth_idx in range(num_depth):
        for y in range(inp_height):
            y_start = y*stride
            y_end = y_start + pool_size
            for x in range(inp_width):
                x_start = x*stride
                x_end = x_start + pool_size

                pool_map[y_start:y_end,
                         x_start:x_end,
                         depth_idx] *= inp[y, x, depth_idx]
    return pool_map


def buildup(inp: np.ndarray, target_shape: tuple):
    '''
    Rebuilds a 4D Tensor from a 2D array

    Parameters:
    - inp (2D array): Input to be reshaped (nodes, examples)
    - target_shape (tuple): Specifies the size of the output (height, width, depth, examples)

    Returns:
    - (4D Tensor) Built up tensor 
    '''
    return inp.reshape((target_shape), order="C")


def _backward(dz, x, k):
    '''
    Performs back prop

    Parameters:
    - dz (3D array): Loss gradient w.r.t pre-acitvaion of the output (height, width, output depth)
    - x (3D array): Input of the layer (height, widht, input depth)
    - k (4D array): Kernels used in the layer (height, widht, input depth, output depth)

    Returns:
    - dk (4D array): Derivative of the Loss w.r.t the kernels
    - db (3D array): Derivative of the Loss w.r.t the biases
    - dx (3D array): Derivative of the Loss w.r.t the input

    Explanation:
    *full explanation is in notes*
    '''

    db = dz
    dk = np.zeros((k.shape))
    for depth_idx in range(dz.shape[2]):
        for channel in range(x.shape[2]):
            dk[:, :, channel, depth_idx] = correlate2d(x[:, :, channel],
                                                       dz[:, :, depth_idx],
                                                       mode="valid")

    dx = np.zeros(x.shape)
    for channel in range(x.shape[2]):
        for depth_idx in range(dz.shape[2]):
            k180 = np.flipud(np.fliplr(k[:, :, channel, depth_idx]))

            dx[:, :, channel] += correlate2d(dz[:, :, depth_idx],
                                             k180, mode="full")

    return dk, db, dx


def save_params(kernels: list, biases: list, file_path: str):
    '''
    Saves Paramaters in a .npz file in the given file path

    Parameters:
    - kernel (list[np.ndarray]): kernels used in the CNN
    - biases (list[np.ndarray]): Biases used in the MLP
    - file_path (str): path to be saved to
    '''

    def check_valid_folders():
        '''
        Checks if file path exists, if it doesnt then creates it
        '''
        if not os.path.exists(file_path):
            os.mkdir(file_path)

    def clear_folder():
        '''
        Clears folder before populating it, 
            (This was done beacuse if there were weights and biases saved before 
            that had 3 layers, then saving weights and biases with 2 layers would 
            not overwrite the 3rd layer, and when loading the weights and biases 
            later would cause errors)
        '''

        files = os.listdir(file_path)

        for file_name in files:
            os.remove(f"{file_path}/{file_name}")

    check_valid_folders()
    clear_folder()

    num_layers = len(kernels)

    for layer in range(num_layers):
        np.save(f"{file_path}/kernels-{layer}.npy", kernels[layer])
        np.save(f"{file_path}/biases-{layer}.npy", biases[layer])


def load_params(file_path: str):
    '''
    Saves Paramaters in a .npy file in the given file path

    Parameters:
    - file_path (str): path to be saved to

    Returns:
    - w (list[2D array])
    - b (list[1D array])

    Raises:
    ValueError: If the file path cannot be found
    '''

    def get_layers():
        '''
        fetches a list of the files and counts the amount of files to get the number of layers

        Raises:
        - Value Error if there are no files in the path
        '''
        files_list = os.listdir(file_path)
        num_layers = len(files_list)//2

        if num_layers <= 0:
            raise ValueError(f"Their are no files in '{file_path}'")
        return num_layers

    if not os.path.exists(file_path):
        raise ValueError(f"File path '{file_path}' does not exist")

    num_layers = get_layers()

    k = []
    b = []
    for layer in range(num_layers):

        k.append(np.load(f"{file_path}/kernels-{layer}.npy"))
        b.append(np.load(f"{file_path}/biases-{layer}.npy"))

    return k, b


def main():
    pass


if __name__ == "__main__":
    main()
