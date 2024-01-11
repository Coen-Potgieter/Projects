
from PIL import Image
import numpy as np
import time
import sys


def euclid_dist(centers, pts):
    '''
    Very pretty function here,

    We are calculating the euclidian distances between every center and every 
    point, this gives us the distances between every point and center in `dists`
    We then find the min of these distances w.r.t every center and this is our 
    index for the smallest distance, we then return this.
    The vectorization here is quite nice

    Parameters:
    - centers (2D array): (center_num, coords)
    - pts (2D array): (point_num, coords)


    Returns:
    - (1D array): index=point index, val=cluster it belongs to

    Example:
    - Say we have 4 pts and 2 clusters
    - say pt 2 is closest to cluster 1
    - then returned is [x, 0, x, x]
    '''

    num_pts = pts.shape[0]
    num_k = centers.shape[0]

    tiled_pts = np.tile(pts, (num_k, 1, 1))
    tiled_centers = np.tile(centers[:, np.newaxis, :], (1, num_pts, 1))
    dists = np.sqrt(np.sum(np.square(tiled_pts - tiled_centers), axis=2))
    return np.argmin(dists, axis=0)


def find_mean(centers, dist_indicies, pts):
    '''
    - centers (2D array): (center_num, coords)
    - dist_indicies (1D array): index=point index, val=cluster it belongs to
    - pts (2D array): random points (point_num, coords)

    Note: 
    - decided to use a for loop becuase sizes of points in each cluster are
        varying
    '''

    num_k = centers.shape[0]
    delta = np.zeros((num_k))
    for c in range(num_k):
        index_pts = np.where(dist_indicies == c)[0]
        if not index_pts.size == 0:
            mean_coords = np.mean(
                pts[np.where(dist_indicies == c)[0]], axis=0)
            delta[c] = np.sum((centers[c, :] - mean_coords)**2)
            centers[c, :] = mean_coords

    return delta


def show_pic_big(pic, scale_factor):
    pic.resize((int(pic.size[0]*scale_factor),
               int(pic.size[1]*scale_factor))).show()
    pass


def get_pixels_RGB(pic):
    return np.array(pic.getdata())


def build_new_pic(pic, centers):

    dims = centers.shape[1]
    width, height = pic.size
    old_pixels = get_pixels_RGB(pic)
    cluster_idx = euclid_dist(centers=centers,
                              pts=old_pixels)

    # crazy indexing things here
    new_pic = centers[cluster_idx, :].reshape(height, width, dims)
    return Image.fromarray(np.uint8(new_pic))


def main():

    num_k = 2
    num_iters = 10
    dims = 3

    pic1 = "Assets/flower.jpeg"
    pic2 = "Assets/dog.png"
    pic3 = "Assets/blackHole.jpeg"
    pic4 = "Assets/tree.jpeg"
    pic = Image.open(pic2).convert("RGB")

    pixel_vals = get_pixels_RGB(pic)
    centers = np.random.uniform(low=0, high=255, size=(num_k, dims))

    delta = np.ones((1))
    i = 0
    k_start = time.time()
    while (np.sum(delta) != 0) and (i <= num_iters):
        dist_indicies = euclid_dist(centers, pixel_vals)
        delta = find_mean(centers, dist_indicies, pixel_vals)
        i += 1
        print(f"Iteration: {i}")

    k_end = time.time()
    print(f"k-means took {k_end - k_start}s")

    centers = centers.astype(np.uint8)
    build_start = time.time()
    new_pic = build_new_pic(pic, centers)
    build_end = time.time()
    print(f"Building image took {build_end- build_start}s")

    show_pic_big(pic, 2)
    show_pic_big(new_pic, 2)


if __name__ == "__main__":
    main()
