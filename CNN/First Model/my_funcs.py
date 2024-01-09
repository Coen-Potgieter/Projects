import numpy as np
from PIL import Image


def my_convolute(img_arr: np.ndarray, kernel: np.ndarray):
    '''
    Performs Valid Cross Correlation for a 2D Array and a sqaure kernel

    Parameters:
    - img_arr (2D array): Input size, (height, width)
    - kernel (2D array): This kernel should be sqaure, (height, width)

    Returns:
    - out (2D array)
    '''

    kernel_size = kernel.shape[0]

    out_height = img_arr.shape[0] - kernel_size + 1
    out_width = img_arr.shape[1] - kernel_size + 1
    out = np.zeros((out_height, out_width))

    for y in range(out_height):
        for x in range(out_width):
            extracted_inp = img_arr[y:y+kernel_size, x:x+kernel_size]
            out[y, x] = np.sum(extracted_inp*kernel)
    return out


def my_pool(img_arr: tuple, pool_size: int, pool_type: str, stride=None):
    """
    Perform pooling on a 2D input array.

    Parameters:
    - img_arr (tuple): Input size, (height, width)
    - pool_size (int): Size of the sqaure pooling window
    - pool_type (str): Pooling type, ("max", "mean", "min")
    - stride (int): Stride. If None, it defaults to pool_size.

    Returns:
    - out (2D array): (height, width)
    """

    valid_pool_types = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean
    }
    if pool_type not in valid_pool_types.keys():
        raise ValueError("Invalid Pool Type")

    def pool(inp):
        return valid_pool_types[pool_type](inp)

    if stride is None:
        stride = pool_size

    out_height = img_arr.shape[0] // pool_size
    out_width = img_arr.shape[1] // pool_size

    out = np.zeros((out_height, out_width))
    for y in range(out_height):
        for x in range(out_width):
            extracted_inp = img_arr[y*stride:y*stride + pool_size,
                                    x*stride:x*stride + pool_size]
            out[y, x] = pool(extracted_inp)
    return out


def my_upscale(img_arr, scale_factor):
    """
    Upscales a 2D array given a scale factor

    Parameters:
    - img_arr (2D array): Input, (height, width)
    - scale_factor (int)

    Returns:
    - out (2D array)
    """

    height = img_arr.shape[0]
    width = img_arr.shape[1]
    out = np.zeros((height*scale_factor, width*scale_factor))
    for y in range(height):
        out_y = y*scale_factor
        for x in range(width):
            out_x = x * scale_factor
            out[out_y:out_y + scale_factor,
                out_x:out_x+scale_factor] = img_arr[y, x]
    return out


def apply_activation(inp: np.ndarray, act_func: str):

    def ReLU(x):
        return np.maximum(x, 0)

    def hyp_tan(x):
        return np.tanh(x)

    def leaky_ReLU(x, alpha=0.01):
        return np.maximum(alpha * x, x)

    def der_leaky_ReLU(x, alpha=0.01):
        return np.where(x >= 0, 1, alpha)

    valid_funcs = {
        "ReLU": ReLU,
        "Tanh": hyp_tan,
        "Leaky ReLU": leaky_ReLU,
    }

    if act_func not in valid_funcs.keys():
        raise ValueError(
            f"Invalid Activation Function `{act_func}`, Valid Functions are{list(valid_funcs.keys())}")

    return valid_funcs[act_func](inp)


def img_load(file):
    img = Image.open(file)
    return img


def main():

    def test_Pool_Upscale(pool_size):
        depth = 3
        out_mat = []
        for channel in range(depth):
            ph_mat = my_pool(img_mat[:, :, channel],
                             pool_size, pool_type="max")
            out_mat.append(my_upscale(ph_mat, scale_factor=pool_size))

        out = np.stack(out_mat, axis=-1)
        out_img = Image.fromarray(np.uint8(out))

        img.show()
        out_img.show()
        pass

    def test_Convolute(size):
        '''Blur'''
        depth = 3
        kernel = np.full((size, size), 1/(size**2))

        out = []
        for channel in range(depth):
            out.append(my_convolute(img_mat[:, :, channel], kernel))

        out = np.stack(out, axis=-1)
        out_img = Image.fromarray(np.uint8(out))
        img.show()
        out_img.show()

    img_paths = [
        "Assets/FunctionTest/200pix.png",
        "Assets/FunctionTest/600pix.png"
    ]
    img = img_load(img_paths[1])
    img_mat = np.array(img)

    test_Pool_Upscale(10)
    test_Convolute(10)


if __name__ == "__main__":
    main()

    pass
