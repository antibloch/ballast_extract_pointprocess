import open3d as o3d
import numpy as np

file_path = "scan.ply"

# Read the point cloud as a tensor
pcd_np = o3d.t.io.read_point_cloud(file_path)


original_pcd = o3d.geometry.PointCloud()
original_pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
original_pcd.colors = o3d.utility.Vector3dVector(pcd_np[:, 3:6])

# Rotating point cloud
# R_x = original_pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))  # Rotate 90° around X-axis
# R_y = original_pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))  # Rotate 90° around Y-axis
# R_z = original_pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))  # Rotate 90° around Z-axis
# original_pcd.rotate(R_x, center=(0, 0, 0))  # Apply the X rotation
# o3d.visualization.draw_geometries([original_pcd])


plane_model, inliers = original_pcd.segment_plane(distance_threshold=0.2,  # Adjust based on your data scale
                                        ransac_n=3,
                                        num_iterations=1000)
a, b, c, d = plane_model
print(f"Detected ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
# Compute the normalization factor for the plane normal
norm_factor = np.sqrt(a**2 + b**2 + c**2)
# Convert the point cloud to a NumPy array of points
pcd_np = np.asarray(original_pcd.points)
# Compute the perpendicular distance from each point to the ground plane
distances = np.abs(a * pcd_np[:, 0] + b * pcd_np[:, 1] + c * pcd_np[:, 2] + d) / norm_factor
# Normalize distances to the range [0, 1] for color mapping
min_dist = np.min(distances)
max_dist = np.max(distances)
normalized_intensity = (distances - min_dist) / (max_dist - min_dist)
# Map the normalized intensity to a color channel.
# Here, we map the intensity to the red channel while keeping green and blue zero.
colors = np.stack([normalized_intensity, np.zeros_like(normalized_intensity), np.zeros_like(normalized_intensity)], axis=1)
# Create a new point cloud that encodes distance from ground as color intensity
pcd_with_distance = o3d.geometry.PointCloud()
pcd_with_distance.points = o3d.utility.Vector3dVector(pcd_np)  # Use the same points
pcd_with_distance.colors = o3d.utility.Vector3dVector(colors)   # Apply our computed colors
# Visualize the point cloud
o3d.visualization.draw_geometries([pcd_with_distance])