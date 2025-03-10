import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def cloud_to_image(pcd_np, resolution):
    minx = np.min(pcd_np[:, 0])
    maxx = np.max(pcd_np[:, 0])
    miny = np.min(pcd_np[:, 1])
    maxy = np.max(pcd_np[:, 1])
    print(f"Point Cloud Bounds: X[{minx}, {maxx}], Y[{miny}, {maxy}]")
    
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    print(f"Computed Image Dimensions: Width={width}, Height={height}")
    
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for point in pcd_np:
        x, y, *_ = point
        r, g, b = point[-3:]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        
        # Ensure the pixel coordinates are within image bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            image[pixel_y, pixel_x] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return image



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


image = cloud_to_image(filtered_positions, resolution=0.1)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()