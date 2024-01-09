import numpy as np
import scipy.signal
import time

def time_func(function):
    def wrapper(*args):
        start = time.time()
        outp = function(args[0], args[1])
        end = time.time()
        print(f"{function.__name__} took {round((end-start)*1000, 2)}ms to execute")
        return outp
    return wrapper

@time_func
def my_convolve1(img, kernel):
    '''Attempt 1, only takes in kernels of odd size'''

    kernel_size = kernel.shape[0]
    radius = kernel_size // 2
    kernel_map = [(x, y) for y in range(-radius, radius+1)
                  for x in range(-radius, radius+1)]

    width, height = img.shape[0], img.shape[1]

    

    out = np.zeros((height - kernel_size + 1, width - kernel_size + 1))
    
    out_index = 0
    for x in range(radius, width - radius):
        
        for y in range(radius, height - radius):
            new_col = 0
            for idx, pixel_map in enumerate(kernel_map):
                old_pixel = img[y+pixel_map[1], x+pixel_map[0]]

                new_col = new_col + old_pixel * \
                    kernel.flatten()[idx]

            out[out_index % out.shape[0], out_index //
                out.shape[1]] = new_col
            out_index += 1
    return out
        
@time_func
def lib_convolve(img_arr, kernel):
    return scipy.signal.convolve2d(img_arr, kernel, mode="valid")
    

@time_func
def fft_convolve(img_arr, kernel):
    return scipy.signal.fftconvolve(img_arr, kernel, mode="valid")

@time_func
def my_convolve2(img_arr, kernel):
    '''
    The mode for this convolution is `Valid` 
    
    It will take in a single img_arr matrix and convolute it with a single 
    kernel matrix, Note: kernel shoudl be sqaure
    All matrices -> (height, width)
    '''
    
    kernel_size = kernel.shape[0]

    out_height = img_arr.shape[0] - kernel_size + 1
    out_width = img_arr.shape[1] - kernel_size + 1
    out = np.zeros((out_height, out_width))

    for y in range(out_height):
        for x in range(out_width):
            extracted_inp = img_arr[y:y+kernel_size, x:x+kernel_size]
            out[y,x] = np.sum(extracted_inp*kernel)
    return out

from numpy.lib.stride_tricks import as_strided
@time_func
def GPT_convolve(img_arr, kernel):
    kernel_size = kernel.shape[0]
    out_height, out_width = img_arr.shape[0] - kernel_size + 1, img_arr.shape[1] - kernel_size + 1

    # Create a view on the input array without copying data
    img_view = as_strided(img_arr, shape=(out_height, out_width, kernel_size, kernel_size), strides=img_arr.strides * 2)

    # Perform element-wise multiplication and summation using NumPy operations
    out = np.einsum('ijkl,kl->ij', img_view, kernel)

    return out

from numpy.lib.stride_tricks import sliding_window_view
def my_convolute3(img_arr, kernel):
    '''
    Gonna try to use numpy stride things and allow it to do multiple examples
    going to try allow it be done with more than 2d arrays

    in CNNs we have (height, width, channel, depth)
    again,
        channel is like RGB
        depth is number of kernels that this single channel must go through
    
    - img_arr (3D Tensor): (height, width, channel)

    - going to try using stride_tricks.sliding_window_view
    '''
    i, j = np.ogrid[:3, :4]
    x = 10*i + j
    idk = sliding_window_view(x=x, window_shape=(2,2))

    print(x)
    for elem in range(idk.shape[0]):
        print(idk[elem,:,:,:])
    
    # print(test1.strides)
    pass

def main():
    inp3x3 = np.array([[1,6,2], 
                    [5,3,1],
                    [7,0,4]])
    
    inp5x5 = np.random.randn(5,5)
    inp10x10 = np.random.randn(10,10)
    inp100x100 = np.random.randn(100,100)
    inp1000x1000 = np.random.randn(1000,1000)
    
    inp20x100 = np.random.randn(20,100)

    kernel2x2 = np.array([[1,2],
                       [-1,0]])
    kernel3x3 = np.random.rand(3,3)
    
    inp = inp1000x1000
    kernel = kernel2x2

    my_convolute3(inp, kernel)
    return
    lib_convolved = lib_convolve(inp, kernel)
    fft_convolved = fft_convolve(inp, kernel)
    gpt_convolved = GPT_convolve(inp, np.flipud(np.fliplr(kernel)))

    # flipped_kernel = np.flipud(np.fliplr(kernel))

    # Note: convolve1 must have a kernel of odd size (3,5,7,...) and a sqaure input
    my1_convolved = np.ones(lib_convolved.shape)
    # my1_convolved = my_convolve1(inp, np.flipud(np.fliplr(kernel)))

    my2_convolved = my_convolve2(inp, np.flipud(np.fliplr(kernel)))

    print()
    if np.allclose(lib_convolved, gpt_convolved):
        print("gpt works")
    else:
        print("gpt Does not work")

    if np.allclose(lib_convolved, my1_convolved):
        print("Convolve 1 works")
    else:
        print("Convolve 1 Does not work")
    
    if np.allclose(lib_convolved, my2_convolved):
        print("Convolve 2 works")
    else:
        print("Convolve 2 Does not work")






if __name__ == "__main__":
    main()