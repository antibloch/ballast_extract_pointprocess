import open3d as o3d
import open3d.core as o3c
import numpy as np

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
mask = (classes == 9) | (classes == 0)
filtered_positions = positions[np.squeeze(mask,-1)]
filtered_colors = filtered_colors[np.squeeze(mask,-1)]

# Create a new Open3D point cloud with only class 2
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_positions)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Save and visualize
o3d.io.write_point_cloud("filtered_class_2.ply", filtered_pcd)
o3d.visualization.draw_geometries([filtered_pcd])
