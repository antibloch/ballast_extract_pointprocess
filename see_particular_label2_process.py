import open3d as o3d
import open3d.core as o3c
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



# ref: https://github.com/WHU-USI3DV/WHU-Railway3D/blob/master/repos/KPConv-PyTorch/datasets/Railway3D.py

# 0:  'rails',
# 1:  'track bed',
# 2:  'masts',
# 3:  'support devices',
# 4:  'overhead lines',
# 5:  'fences',
# 6:  'poles',
# 7:  'vegetation',
# 8:  'buildings',
# 9:  'ground', 
# 10: 'other'


file_path = "scan.ply"

# Read the point cloud as a tensor
pcd = o3d.t.io.read_point_cloud(file_path)

# Convert to numpy arrays
positions = pcd.point.positions.numpy()
if "class" in pcd.point:
    classes = pcd.point["class"].numpy()
else:
    raise ValueError("The 'class' attribute is missing in the PLY file.")

# Print shapes to debug
print("Positions shape before reshape:", positions.shape)
print("Classes shape:", classes.shape)

# Find unique classes
unique_classes = np.unique(classes)
print("Unique classes in the point cloud:", unique_classes)

# Generate a random color map for classes
colors = np.random.rand(len(unique_classes), 3)  # RGB values between 0-1

# Assign colors based on class values
filtered_colors = np.array([colors[np.where(unique_classes == c)[0][0]] for c in classes])

# Convert filtered_colors to Open3D tensor
pcd.point.colors = o3c.Tensor(filtered_colors, dtype=o3c.float32, device=pcd.device)



# sample class==2
mask = classes == 1
filtered_positions = positions[np.squeeze(mask,-1)]
filtered_colors = filtered_colors[np.squeeze(mask,-1)]

# Create a new Open3D point cloud with only class 2
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_positions)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save and visualize
o3d.io.write_point_cloud("filtered_class_2.ply", filtered_pcd)
# o3d.visualization.draw_geometries([filtered_pcd])



# # -------------------------------------DBSCAN (Open3D)----------------------------------------------
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         filtered_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# filtered_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([filtered_pcd],
#                                   zoom=0.455,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])
# # -------------------------------------DBSCAN (Open3D)----------------------------------------------


filtered_positions = torch.tensor(filtered_positions, dtype=torch.float32).to(torch.device("cuda"))
# downsample filtered_positions
filtered_positions = filtered_positions[::10]

# -------------------------------------DBSCAN----------------------------------------------
# dbscan = DBSCAN(eps=0.1, min_samples=100)  # Adjust `eps` based on your data
# labels = dbscan.fit_predict(filtered_positions.cpu().numpy())
# classes = np.expand_dims(labels,-1)
# positions = filtered_positions.cpu().numpy()
# -------------------------------------DBSCAN----------------------------------------------


# -------------------------------------KMEANS----------------------------------------------
def kmeans_pytorch(X,num_clusters, num_iterations = 100, tol = 1e-4):
    N, D = X.shape

    C = X[torch.randperm(N)[:num_clusters]] # (num_clusters, D)
    for i in range(num_iterations):
        # (N, num_clusters)
        dists = torch.cdist(X, C)
        # (N,)
        labels = dists.argmin(dim=1)
        new_C = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])
        if torch.allclose(C, new_C, tol):
            break
        C = new_C

    return labels


labels = kmeans_pytorch(filtered_positions, 500)
classes = np.expand_dims(labels.cpu().numpy(),-1)
positions = filtered_positions.cpu().numpy()
# -------------------------------------KMEANS----------------------------------------------

unique_classes = np.unique(classes)
print("Unique classes in the point cloud:", unique_classes)

# Generate a random color map for classes
colors = np.random.rand(len(unique_classes), 3)  # RGB values between 0-1

# Assign colors based on class values
filtered_colors = np.array([colors[np.where(unique_classes == c)[0][0]] for c in classes])

filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(positions)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Visualize
o3d.visualization.draw([filtered_pcd ])

