from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys

def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def class_pts(rand_pts, centers):

    num_clusters = centers.shape[0]
    clusters = [np.zeros((1, 2)) for _ in range(num_clusters)]
    distances = np.zeros((num_clusters,))
    for point_idx in range(rand_pts.shape[0]):
        for k in range(num_clusters):
            distances[k] = dist(rand_pts[point_idx, :],centers[k, :])
        
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx] = np.append(clusters[cluster_idx], 
                            rand_pts[point_idx:point_idx+1,:], 
                            axis=0)
    return clusters

    

def mean_pts(clusters, centers):
    
    for k in range(len(clusters)):
        centers[k,:] = np.mean(clusters[k], axis=0)

def main():
    
    def anim(i):
        plt.clf()

        for k in range(centers.shape[0]):
            plt.scatter(x=centers[k,0],
                        y=centers[k,1],
                        c=cols[k], 
                        s=300)
        
        clusters = class_pts(rand_points, centers)
        mean_pts(clusters, centers)
        
        for cluster_idx in range(len(clusters)):
            plt.scatter(x=clusters[cluster_idx][:,0],
                        y=clusters[cluster_idx][:,1],
                        c=cols[cluster_idx])
            
    num_points = 300
    num_k = 5
    ms_delay = 500

    cols = [np.random.uniform(low=0, high=1, size=(1,3)) for _ in range(num_k)]

    rand_points = np.random.randint(0,100, size=(num_points,2))

    plt.style.use("fivethirtyeight")
    centers = np.random.uniform(low=1, high=100, size=(num_k,2))
    
    for k in range(centers.shape[0]):
        plt.scatter(x=centers[k,0],
                        y=centers[k,1],
                        c=cols[k], 
                        s=300)
        
    plt.scatter(x=rand_points[:,0], y=rand_points[:,1],
                c="grey")

    ani = FuncAnimation(plt.gcf(), anim, interval=ms_delay)
    
    plt.show()

if __name__ == "__main__":
    main()