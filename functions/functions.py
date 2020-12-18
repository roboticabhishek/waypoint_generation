

# ____________________________________________________________________________________________________________________________
# IMPORTS
# ____________________________________________________________________________________________________________________________

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale



# ____________________________________________________________________________________________________________________________
# HELPER FUNCTIONS
# ____________________________________________________________________________________________________________________________

"""
Function: get_axes()
Description: A function which creates X, Y, Z axes as meshes to be displayed along with the subject mesh
Input: None
Output: X, Y, Z axes as 3D meshes in red, green, blue colors respectively
"""
def get_axes():
    Ry = (np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    Rx = (np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    x_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=5.0, cone_radius=8.0, cylinder_height=400.0, cone_height=30.0, resolution=20, cylinder_split=4, cone_split=1)
    x_axis.paint_uniform_color((np.array([1, 0, 0])))
    x_axis.rotate(Rx, (np.array([0, 0, 0])))
    y_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=5.0, cone_radius=8.0, cylinder_height=400.0, cone_height=30.0, resolution=20, cylinder_split=4, cone_split=1)
    y_axis.rotate(Ry, (np.array([0, 0, 0])))
    y_axis.paint_uniform_color((np.array([0, 1, 0])))
    z_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=5.0, cone_radius=8.0, cylinder_height=400.0, cone_height=30.0, resolution=20, cylinder_split=4, cone_split=1)
    z_axis.paint_uniform_color((np.array([0, 0, 1])))
    return x_axis, y_axis, z_axis




"""
Function: Kmeans_with_evaluation()
Description: A function which clusters the 6D data and checks if each cluster is within camera focus
Input: init_method -> Kmeans parameter (see scikit-learn), 
       num_clusters -> Kmeans parameter (see scikit-learn), 
       num_runs -> Kmeans parameter (see scikit-learn), 
       max_iter -> Kmeans parameter (see scikit-learn), 
       data -> feature matrix, 
       pcd -> pointcloud, 
       bbcheck_scale -> scaling factor for bounding box, useful for flat geometries
Output: The area and depth of the biggest cluster formed
"""
def Kmeans_with_evaluation(init_method, num_clusters, num_runs, max_iter, data, pcd, bbcheck_scale):
    KM = KMeans(init=init_method, n_clusters=num_clusters, n_init=num_runs, max_iter=max_iter)
    KM.fit(data)
    labels = KM.labels_
    cluster_collection = [[] for i in range(num_clusters)]
    for j in range(len(labels)):
        cluster_collection[labels[j]].append(j)
    extents = []
    spans = []
    for i in range(num_clusters):
        temp_pcd = pcd.select_by_index(cluster_collection[i])
        c = temp_pcd.get_center()
        temp_pcd.scale(bbcheck_scale, c)
        temp_pcd_hull, _ = temp_pcd.compute_convex_hull()
        temp_bb = temp_pcd_hull.get_oriented_bounding_box()
        temp_bb_extents = temp_bb.extent
        extents.append(temp_bb_extents[2])
        spans.append(temp_bb_extents[0]*temp_bb_extents[1])
    return (max(extents))/bbcheck_scale, (max(spans))/(bbcheck_scale*bbcheck_scale)




"""
Function: range_checker()
Description: Checkes whether a given waypoint is to be filtered or not
Input: point -> 3D point,
       extents_for_filter -> thresholds for filtering (calculated automatically),
       reject_list -> a list which indicated which axes' waypoints to filter out       
Output: boolean
"""
def range_checker(point, extents_for_filter, reject_list):
    for i in range(6):
        if (reject_list[i] == 1):
            if (i<3):
                if (point[(i)] > extents_for_filter[(i)]):
                    return False
            else:
                if (point[(i%3)] < -extents_for_filter[(i%3)]):
                    return False
    return True