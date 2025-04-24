from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys

BLACK = "#000000"


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
    - (1D array): index=center index, val=point index

    Example:
    if point 12 is closest to center 2 then returned will be [x,11,x,....]
    '''

    num_pts = pts.shape[0]
    num_k = centers.shape[0]
    tiled_pts = np.tile(pts, (num_k, 1, 1))
    tiled_centers = np.tile(centers[:, np.newaxis, :], (1, num_pts, 1))

    dists = np.sqrt(np.sum(np.square(tiled_centers - tiled_pts), axis=2))
    return np.argmin(dists, axis=0)


def find_mean(centers, dist_indicies, pts):
    '''
    - centers (2D array): (center_num, coords)
    - dist_indicies (1D array): (points), vals=center_index
    - pts (2D array): random points (point_num, coords)

    end up with (4,3)

    (num_k, num_pts, coords)
    '''

    num_k = centers.shape[0]
    for c in range(num_k):
        index_pts = np.where(dist_indicies == c)[0]
        if not index_pts.size == 0:
            centers[c, :] = np.mean(
                pts[np.where(dist_indicies == c)[0]], axis=0)
        else:
            centers[c, :] = np.random.randint(low=0, high=10, size=(3,))


def main():
    def figure_config():
        fig = plt.figure()
        fig.set_size_inches(w=12, h=6)
        fig.set_facecolor(BLACK)
        return fig

    def axis_config():
        axes = fig.add_subplot(111, projection='3d')
        axes.set_facecolor(BLACK)
        axes.xaxis.pane.set_facecolor(BLACK)
        axes.yaxis.pane.set_facecolor(BLACK)
        axes.zaxis.pane.set_facecolor(BLACK)

        axes.xaxis._axinfo["grid"]['color'] = BLACK
        axes.yaxis._axinfo["grid"]['color'] = BLACK
        axes.zaxis._axinfo["grid"]['color'] = BLACK

        axes.xaxis.pane.set_edgecolor(BLACK)
        axes.yaxis.pane.set_edgecolor(BLACK)
        axes.zaxis.pane.set_edgecolor(BLACK)
        return axes

    def turn_anim(i):
        nonlocal initial_turn_angle
        # sets angle
        axes.view_init(elev=30, azim=initial_turn_angle, roll=0)

        # increments angle
        initial_turn_angle += turn_speed

    def cluster_update_anim(i):
        nonlocal plot_pts, plot_centers

        # removing previously plotted points
        for elem in plot_pts:
            elem.remove()
        plot_centers.remove()

        # perform 1 step of k-clusters
        dist_indicies = euclid_dist(centers, pts)
        find_mean(centers, dist_indicies, pts)
        classed_pts = [pts[np.where(dist_indicies == c)[0]]
                       for c in range(centers.shape[0])]
        
        # plotting
        plot_pts = []
        for c in range(num_k):
            plot_pts.append(axes.scatter(classed_pts[c][:, 0],
                                         classed_pts[c][:, 1],
                                         classed_pts[c][:, 2],
                                         c=cols[c], marker="o", s=20, alpha=1))
        plot_centers = axes.scatter(centers[:, 0],
                                    centers[:, 1],
                                    centers[:, 2],
                                    c=cols, marker="o", s=300, alpha=1)

    # cluster settings
    num_points = 500
    num_k = 3
    ms_delay = 500

    # graph settings
    turn_speed = 0.3
    initial_turn_angle = 0

    # setting cols for each cluster
    cols = [np.random.uniform(low=0, high=1, size=(1, 3))
            for _ in range(num_k)]
    # random points
    pts = np.random.randint(low=0, high=10, size=(num_points, 3))
    # random clusters
    centers = np.random.randint(low=0, high=10, size=(num_k, 3))

    plt.style.use("fivethirtyeight")

    fig = figure_config()
    axes = axis_config()

    # plotting
    plot_pts = [axes.scatter(pts[:, 0],
                            pts[:, 1],
                            pts[:, 2],
                            c="grey", marker="o", s=20, alpha=1)]

    plot_centers = axes.scatter(centers[:, 0],
                                centers[:, 1],
                                centers[:, 2],
                                c=cols, marker="o", s=300, alpha=1)
    
    # animations
    anim1 = FuncAnimation(plt.gcf(), turn_anim, interval=1)
    anim2 = FuncAnimation(plt.gcf(), cluster_update_anim, interval=ms_delay)

    plt.show()


if __name__ == "__main__":
    main()
