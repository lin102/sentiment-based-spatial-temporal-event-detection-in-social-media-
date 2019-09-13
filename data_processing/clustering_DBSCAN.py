from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def clustering_DBSCAN(path, distance, min_sampples_number):

    records = pd.read_csv(path, encoding='utf-8')
    # Earth Radius = 6371.0088
    kms_per_radian = 6371.0088
    # epsilon is the radian unit for km
    epsilon = distance/kms_per_radian
    coordinates = records[['WGS84Latitude','WGS84Longitude']]
    #DBSCAN clustering
    cluster = DBSCAN(eps=epsilon, min_samples=min_sampples_number, algorithm='ball_tree', metric='haversine',n_jobs=-1).fit(np.radians(coordinates))
    # cluster labbel
    cluster_labels = cluster.labels_
    # cluster number include the noise cluster
    num_clusters = len(set(cluster_labels)) - 1
    #print(set(cluster_labels))
    print(num_clusters)
    #print(cluster.core_sample_indices_ )
    #print(cluster.components_)
    coordinates.loc[:, 'label'] = cluster_labels
    #print(coordinates)
    # filter all the noises
    coordinates = coordinates[coordinates.label != -1]

    return coordinates, num_clusters


def draw_cluster_small_multiples(path, distance_range, min_samples_range):
    # --------- Draw cluster samll multiples--------------

    fig, axes = plt.subplots(nrows=3, ncols=3) # 9 small multiples
    # axes is a axis array which is all the subplots
    fig.suptitle('DBSCAN Clustering Results With Different Parameters ')

    # create rainbow colors for clusters
    colors = cm.rainbow(np.linspace(0, 1, 20))

    # draw every subplot (ax)
    for ax, dis, sample in zip(axes.flat, distance_range, min_samples_range):

        coordinates, cluster_number = clustering_DBSCAN(path, dis, sample)

        # draw clusters in different colors
        for cluster_num in range(cluster_number):
            cluster = coordinates[coordinates.label == cluster_num]
            ax.scatter(cluster.WGS84Longitude, cluster.WGS84Latitude, c=colors[cluster_num])

        ax.set_title('Clusters: '+ str(cluster_number)+' Epsilon: '+str(dis)+'km  Min_sample: '+str(sample))

    plt.show()


def export_optimized_clusters(path, distance, min_sampples_number,save_path):
    data = clustering_DBSCAN(path, distance, min_sampples_number)[0]
    data.to_csv(save_path, encoding='utf-8', index=False)
    print("Clusters Exported!")


if __name__ == '__main__':

    path = "../transport_2014.csv"
    # set parameter distance range
    distance_range = [0.5, 1, 1.5, 0.5, 1, 1.5, 0.5, 1, 1.5]  # distance = [0.5, 1, 1.5]
    # set parameter min samples range
    min_samples_range = [5, 5, 5, 10, 10, 10, 15, 15, 15]  # min_samples = [5, 10 ,15]

    draw_cluster_small_multiples(path, distance_range, min_samples_range)

    save_path = '../best_clusters.csv'
    # export the best cluster data
    export_optimized_clusters(path, 1, 15, save_path)
