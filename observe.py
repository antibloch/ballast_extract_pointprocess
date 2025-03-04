import open3d as o3d
import numpy as np

file_path = "hmls_01.ply"

with open(file_path, "r", encoding="latin-1") as file:
    content = file.read()

print("Contents of ply file:")
print(content[:500]) 
print("\n\n")


# Read the point cloud as a tensor (supports attributes like "class")
pcd = o3d.t.io.read_point_cloud(file_path)

# Convert to numpy arrays
positions = pcd.point.positions.numpy()  # x, y, z
if "scalar_Classification" in pcd.point:  # Ensure "class" exists
    classes = pcd.point["scalar_Classification"].numpy()

    print("Unique classes in the point cloud:", np.unique(classes))
else:
    raise ValueError("The 'class' attribute is missing in the PLY file.")
