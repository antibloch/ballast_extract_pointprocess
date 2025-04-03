import open3d as o3d

file_path = "hmls_01.ply"


# Read the PLY file as a point cloud
pcd = o3d.io.read_point_cloud(file_path)

# Check if the file was loaded correctly
if not pcd.is_empty():
    print("Successfully loaded point cloud with", len(pcd.points), "points")
else:
    print("Failed to load point cloud")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

