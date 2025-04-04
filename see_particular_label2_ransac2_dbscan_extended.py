import open3d as o3d
import open3d.core as o3c
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils import *
import random
import time



def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds

    Args:
        points (np.ndarray): 
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations)
    
        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list



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





# downsample filtered_positions
positions = filtered_positions[::10]
colors = filtered_colors[::10]
points = o3d.geometry.PointCloud()
points.points = o3d.utility.Vector3dVector(positions)
points.colors = o3d.utility.Vector3dVector(colors)

points = np.asarray(points.points)

points = RemoveNoiseStatistical(points, nb_neighbors=50, std_ratio=0.5)

#DrawPointCloud(points, color=(0.4, 0.4, 0.4))
t0 = time.time()
results = DetectMultiPlanes(points, min_ratio=0.05, threshold=0.005, iterations=2000)
print('Time:', time.time() - t0)
planes = []
colors = []
for _, plane in results:

    r = random.random()
    g = random.random()
    b = random.random()

    color = np.zeros((plane.shape[0], plane.shape[1]))
    color[:, 0] = r
    color[:, 1] = g
    color[:, 2] = b

    planes.append(plane)
    colors.append(color)

planes = np.concatenate(planes, axis=0)
colors = np.concatenate(colors, axis=0)

features = np.hstack([planes, colors])

features = torch.from_numpy(features).float().to('cuda')



dbscan = DBSCAN(eps=0.15, min_samples=10)  # Adjust `eps` based on your data
labels = dbscan.fit_predict(features.cpu().numpy())
classes = np.expand_dims(labels,-1)
positions = planes

classes = torch.tensor(classes)

unique_classes, counts = torch.unique(classes, return_counts=True)

max_class = unique_classes[counts.argmax()]

mask = classes == max_class

print("Size of Positions:", positions.shape)
filtered_positions = positions[np.squeeze(mask,-1)]
print("Size of filtered_positions:", filtered_positions.shape)
filtered_colors = colors[np.squeeze(mask,-1)]


filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_positions)
# filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Visualize
o3d.visualization.draw([filtered_pcd ])

