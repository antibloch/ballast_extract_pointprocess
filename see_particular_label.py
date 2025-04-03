import open3d as o3d
import numpy as np

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

# Filter for class 2
mask = classes == 9
filtered_positions = positions[np.squeeze(mask,-1)]

# Create a new Open3D point cloud with only class 2
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_positions)


# Save and visualize
o3d.io.write_point_cloud("filtered_class_2.ply", filtered_pcd)
o3d.visualization.draw_geometries([filtered_pcd])

print("Filtered point cloud saved as 'filtered_class_2.ply'.")

