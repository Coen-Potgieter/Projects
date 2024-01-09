from PIL import Image
import numpy as np


def img_load(file):
    img = Image.open(file)
    # img.show()
    return img

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

def my_upscale(img_arr, scale_factor):
    """
    Upscales a 2D array given a scale factor

    Parameters:
    - img_arr: 2D array (e.g., output of a convolutional layer)
    - scale_factor: int that will generally be equal to the pool size

    Returns:
    - out: 2D array after upscaling
    """
        
    height = img_arr.shape[0]
    width = img_arr.shape[1]
    out = np.zeros((height*scale_factor, width*scale_factor))
    for y in range(height):
        out_y = y*scale_factor
        for x in range(width):
            out_x = x * scale_factor
            out[out_y:out_y + scale_factor, out_x:out_x+scale_factor] = img_arr[y,x]
    return out


def unpool1(inp: np.ndarray, pool_map: np.ndarray, pool_size: int, stride=None):

    # idea 1 do sliding windows again, this feels slow and against the point
    #   of having a map in the first place

    # idea 2 expand the pool to the same size as the map and element wise multiply
    #   This feels good
    
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
                         depth_idx] *= inp[y,x,depth_idx]
    return pool_map

def unpool2(inp: np.ndarray, pool_map: np.ndarray, pool_size: int, stride=None):

    # idea 1 do sliding windows again, this feels slow and against the point
    #   of having a map in the first place

    # idea 2 expand the pool to the same size as the map and element wise multiply
    #   This feels good
    
    if stride is None:
        stride = pool_size

    inp_height, inp_width, num_depth = inp.shape
    print(pool_map.shape[0] // inp.shape[0])
    
    
def main():
    img_paths = [
        "Assets/200pix.png",
        "Assets/test-image.png"
    ]
    img = img_load(img_paths[1])

    
    img_mat = np.array(img)

    inp = np.array([[1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16],])
    

    out_mat = my_pool(img_mat, pool_size=10, pool_type="max")

    scaled_mat = my_upscale(out_mat, scale_factor=10)
    scaled_img = Image.fromarray(scaled_mat)

    img.show()
    scaled_img.show()

    # scaled_img = img.resize((300, 300), Image.NEAREST)
    # out_img.resize((img_mat.shape[0], img_mat.shape[1]), Image.NEAREST).show()




if __name__ == "__main__":
    main()