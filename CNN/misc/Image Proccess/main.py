from PIL import Image
import numpy as np
import scipy.signal
import time



def lib_convolve(img, kernel):
    depth = img.shape[2]
    out = []
    for idx in range(depth):
        out.append(scipy.signal.convolve2d(img[:,:,idx], kernel[:,:,idx], mode="valid"))
        
    return np.stack(out,axis=-1)

def my_convolve2(img, kernel):

    def my_convolute2(img_arr, kernel):
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
    
    depth = img.shape[2]
    out = []
    for channel in range(depth):
        inp = img[:,:,channel].squeeze()
        kernel_mat = kernel[:,:,channel].squeeze()
        out.append(my_convolute2(inp, kernel_mat))

    return np.stack(out, axis=-1)





    pass




def my_convolve(img, kernel):
    '''Valid'''

    kernel_size = kernel.shape[0]
    radius = kernel_size // 2
    kernel_map = [(x, y) for y in range(-radius, radius+1)
                  for x in range(-radius, radius+1)]

    width, height = img.shape[0], img.shape[1]

    depth = img.shape[2]

    out = np.zeros((height - kernel_size + 1, width - kernel_size + 1, depth))
    perc_done = 0
    for channel in range(depth):
        out_index = 0
        for x in range(radius, width - radius):
            if x % (width/10) == 0:
                perc_done += 10//depth
                print(f"{perc_done}%")

            for y in range(radius, height - radius):
                new_col = 0
                for idx, pixel_map in enumerate(kernel_map):
                    old_pixel = img[y+pixel_map[1], x+pixel_map[0], channel]

                    new_col = new_col + old_pixel * \
                        kernel[:, :, channel].flatten()[idx]

                out[out_index % out.shape[0], out_index //
                    out.shape[1], channel] = min(max(new_col, 0), 255)
                out_index += 1

    return out


def edge_detection(img):
    grayscale_img = img.convert("L")
    img_array = np.expand_dims(np.array(grayscale_img), axis=-1)

    hor_edge_det = np.expand_dims(np.array([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]),
                                  axis=-1)

    ver_edge_det = np.expand_dims(np.array([[1, 2, 1],
                                            [0, 0, 0],
                                            [-1, -2, -1]]),
                                  axis=-1)

    edge_det = np.expand_dims(np.array([[0, 1, 0],
                        [-1, 4, -1],
                        [0, 1, 0]]),
                        axis=-1)
    
    out = my_convolve(img_array, ver_edge_det).squeeze()
    out_img = Image.fromarray(out)
    out_img.show()

    out = my_convolve(img_array, hor_edge_det).squeeze()
    out_img = Image.fromarray(out)
    out_img.show()

    out = my_convolve(img_array, edge_det).squeeze()
    for idx in range(len(out)):
        out[idx] = 255 - out[idx]
    out_img = Image.fromarray(out)
    out_img.show()
    

    pass


def box_blur(img, kernel_size):

    img_array = np.array(img)  # this is now a 3D Tensor (200x200x3)
    # These numpy arrays, (height, width, depth)

    def blur_kernel(size, depth):
        # this creates a group of kernels with all elements being 1/(size**2)
        return np.full((size, size, depth), 1/(size**2))

    # since the input has 3 channels out kernel must have a depth of 3
    kernel = blur_kernel(size=kernel_size, depth=3)


    # my_start = time.time()
    # out = my_convolve(img_array, kernel)
    # my_end = time.time()
    # out_img = Image.fromarray(np.uint8(out))
    # out_img.show()  

    # lib_start = time.time()
    # out = lib_convolve(img_array, kernel)
    # lib_end = time.time()
    # out_img = Image.fromarray(np.uint8(out))
    # out_img.show()
    
    my2_start = time.time()
    out = my_convolve2(img_array, kernel)
    my2_end = time.time()
    out_img = Image.fromarray(np.uint8(out))
    out_img.show()

    # print(f"Library took {lib_end - lib_start}s")
    # print(f"My convolve took {my_end - my_start}s")
    print(f"My convolve2 took {my2_end - my2_start}s")

def my_pool(img_arr, pool_size, pool_type, stride=None):
    """
    Perform pooling on a 2D input array.

    Parameters:
    - img_arr: 2D array (e.g., output of a convolutional layer)
    - pool_size: Integer specifying the size of the sqaure pooling window
    - pool_type: String specifying the pooling type ("max", "mean", "min")
    - stride: Integer specifying the stride. If None, it defaults to pool_size.

    Returns:
    - out: 2D array after pooling
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
            out[y,x] = pool(extracted_inp)
    return out

def down_RGB_size_img(img, pool_size, pool_type):

    img_mat = np.array(img)
    
    out = []
    for channel in range(img_mat.shape[2]):
        out.append(my_pool(img_mat[:,:,channel], pool_size, pool_type))

    out = np.stack(out ,axis=-1)

    
    out_img = Image.fromarray(np.uint8(out))

    out_img.resize((img_mat.shape[0], img_mat.shape[1]), Image.NEAREST).show()


def img_load(file):
    img = Image.open(file)
    img.show()
    return img


def main():
    img_paths = [
        "Assets/200pix.png",
        "Assets/test-image.png"
    ]
    img = img_load(img_paths[1])

    # box_blur(img, 10)
    edge_detection(img)

    # down_RGB_size_img(img, pool_size=5, pool_type="mean")

if __name__ == "__main__":
    main()
