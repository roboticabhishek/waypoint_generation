"""
MAIN ROUTINE
"""





# ______________________________________________________________
# IMPORTS
# ______________________________________________________________

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from functions import functions


# ______________________________________________________________
# CONSTANTS
# ______________________________________________________________

# part_num saved in /mesh
part_num = 1

# the distance of the camera from the surface. determined experimentally
offset = 300

# determine camera field of view. find the area of the circle with the vertical field of view as its diameter
camera_area = 6400

# camera depth of field
cam_dof = 20

# a constant used in clustering.
bbcheck_scale = 100.0

# input axes to reject those waypoints {x, y, z, -x, -y, -z}
# for example, [0, 1, 1, 0, 0, 1] translates to filtering out all waypoints in the Y, Z, -Z direction
reject_list = [0, 0, 0, 0, 0, 0]



# ________________________________________________________________
# OBTAIN AXES TO DRAW ALONG WITH MESH
# ________________________________________________________________

x_axis,  y_axis, z_axis = functions.get_axes()



# ________________________________________________________________
# MESH PREPROCESSING
# ________________________________________________________________

# translate mesh to its centroid
mesh = o3d.io.read_triangle_mesh("D:/Academics/Thesis/macs_waypoint_gen/mesh/part_{}.STL".format(str(part_num)))
mesh.compute_vertex_normals()
mesh.translate(-mesh.get_center())
o3d.io.write_triangle_mesh("D:/Academics/Thesis/macs_waypoint_gen/mesh/part_{}.STL".format(str(part_num)), mesh)

# visualize the mesh with along with the axes
mesh = o3d.io.read_triangle_mesh("D:/Academics/Thesis/macs_waypoint_gen/mesh/part_{}.STL".format(str(part_num)))
mesh.remove_duplicated_vertices()
mesh.compute_vertex_normals()
AABB = mesh.get_axis_aligned_bounding_box()
ET = AABB.get_extent()
o3d.visualization.draw_geometries([mesh, x_axis, y_axis, z_axis])

# resample the mesh to remove symmetry and improve mesh quality of flat geometries
num_vertices = int(mesh.get_surface_area()/100)
throwaway_pcd = mesh.sample_points_poisson_disk(number_of_points=num_vertices)
throwaway_pcd.estimate_normals()
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(throwaway_pcd)
o3d.io.write_triangle_mesh("D:/Academics/Thesis/macs_waypoint_gen/mesh/part_{}_resampled.PLY".format(str(part_num)), mesh)
mesh = o3d.io.read_triangle_mesh("D:/Academics/Thesis/macs_waypoint_gen/mesh/part_{}_resampled.PLY".format(str(part_num)))



# ________________________________________________________________
# DATA COMPILATION
# ________________________________________________________________

# feature scaling. mapping all values of the the part coordinates between (-1, 1)
mesh.compute_vertex_normals()
mesh_normals = np.asarray(mesh.vertex_normals)
mesh_vertices = np.asarray(mesh.vertices)
mesh_vertices = 2 * (minmax_scale(mesh_vertices) - 0.5)

# create feature matrix
data = np.concatenate((mesh_vertices, mesh_normals), axis=1)



# ________________________________________________________________
# K-MEANS IMPLEMENTATION WITH BINARY SEARCH
# ________________________________________________________________

# K means constants
init_method = "random"
max_iter = 100
num_runs = 10

# binary search to find optimal k
pcd = o3d.io.read_point_cloud("D:/Academics/Thesis/macs_waypoint_gen/mesh/part_{}_resampled.PLY".format(str(part_num)))
bs_high = len(pcd.points)//100
bs_mid = 0
bs_low = 1
while(bs_high > bs_low):
    print(bs_low, bs_mid, bs_high)
    bs_mid = (bs_low + bs_high)//2
    max_depth, max_area = functions.Kmeans_with_evaluation(init_method, bs_mid, num_runs, max_iter, data, pcd, bbcheck_scale)
    if (max_depth > 0.8*cam_dof) or (max_area > 0.8*camera_area):
        bs_low = bs_mid + 1
    else:
        bs_high = bs_mid
print("optimal no. of clusters: {}".format(bs_high))

# kmeans with the final number of clusters
num_clusters = bs_high
KM_final = KMeans(init=init_method, n_clusters=num_clusters, n_init=num_runs, max_iter=max_iter)
KM_final.fit(data)
labels = KM_final.labels_


# ________________________________________________________________
# DERIVING DIFFERENT ELEMENTS OF THE MESH
# ________________________________________________________________

# cluster_collection. this is a list of list of labels which classifies points according to clusters
cluster_collection = [[] for i in range(num_clusters)]
for j in range(len(labels)):
    cluster_collection[labels[j]].append(j)

# cluster_collection_color. coloed all clusters for visualization
cluster_collection_color = []
for i in range(num_clusters):
    temp_pcd = pcd.select_by_index(cluster_collection[i])
    color = np.random.random([3, 1])
    temp_pcd.paint_uniform_color(color)
    cluster_collection_color.append(temp_pcd)
# o3d.visualization.draw_geometries(cluster_collection_color)



# ________________________________________________________________
# WAYPOINT GENERATION
# ________________________________________________________________

# mesh_list. list containing segmented meshes that are final subregions
mesh_list = []
for i in range(num_clusters):
    temp_mesh = mesh.select_by_index(cluster_collection[i])
    mesh_list.append(temp_mesh)
# o3d.visualization.draw_geometries(mesh_list, point_show_normal=True)

# waypoints_projection_list, waypoints_normals_list. list of coordinate centers and normals of all subregions
waypoints_projection_list = []
waypoints_normals_list = []
for i in range(len(mesh_list)):
    temp_mesh = mesh_list[i]
    temp_mesh.remove_duplicated_vertices()
    temp_mesh.compute_vertex_normals()
    temp_xyz = np.asarray(temp_mesh.vertices)
    temp_uvw = np.asarray(temp_mesh.vertex_normals)
    temp_xyz = np.mean(temp_xyz, axis=0)
    temp_uvw = np.mean(temp_uvw, axis=0)
    temp_uvw /= np.linalg.norm(temp_uvw)
    waypoints_projection_list.append(temp_xyz)
    waypoints_normals_list.append(temp_uvw)

# calculating the waypoints' 3D coordinates
N = len(waypoints_projection_list)
displacements = [offset*norms for norms in waypoints_normals_list]
WAYPOINTS_XYZ_prefilter = [(waypoints_projection_list[i] + displacements[i]) for i in range(N)]


# ________________________________________________________________
# WAYPOINT FILTERING
# ________________________________________________________________

# thresholds for each axes. based on the part dimensions
extents_for_filter = [np.abs(ET[0]/2), np.abs(ET[1]/2), np.abs(ET[2]/2)]

# filtering waypoints
# WAYPOINTS_XYZ -> list of 3d coordinates of waypoints
# WAYPOINTS_XYZ -> list of corresponding normals
WAYPOINTS_XYZ = []
WAYPOINTS_N = []
for i in range(len(WAYPOINTS_XYZ_prefilter)):
    point = WAYPOINTS_XYZ_prefilter[i]
    allowed = functions.range_checker(point, extents_for_filter, reject_list)
    if (allowed == True):
        WAYPOINTS_XYZ.append(point)
        WAYPOINTS_N.append(waypoints_normals_list[i])
print("no. of waypoints before filtering: {}".format(str(N)))
print("no. of waypoints after filtering: {}".format(str(len(WAYPOINTS_XYZ))))



# ________________________________________________________________
# WAYPOINT VISUALIZATION
# ________________________________________________________________

# convert to pcd and visualize
WAYPOINTS_XYZ_arr = np.array(WAYPOINTS_XYZ)
waypoint_visualization_pcd = o3d.geometry.PointCloud()
waypoint_visualization_pcd.points = o3d.utility.Vector3dVector(WAYPOINTS_XYZ_arr)
black = np.zeros((3, 1))
waypoint_visualization_pcd.paint_uniform_color(black)
# o3d.visualization.draw_geometries([waypoint_visualization_pcd, mesh, x_axis,  y_axis, z_axis])




