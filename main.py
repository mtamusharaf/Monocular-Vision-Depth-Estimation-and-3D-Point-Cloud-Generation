import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# encoders = ['vits', 'vitb', 'vitl']
encoder = 'vits'
video_path = 0

margin_width = 50
caption_height = 60

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

depth_anything.eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def depth_estimation(raw_image):
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    cv2.imwrite('depth_image.png', depth_color)

    split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
    combined_results = cv2.hconcat([raw_image, split_region, depth_color])

    caption_space = np.ones((caption_height, combined_results.shape[1], 3),
dtype=np.uint8) * 255
    captions = ['Raw image', 'Depth Anything']
    segment_width = w + margin_width
    for i, caption in enumerate(captions):
        # Calculate text size
        text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

        # Calculate x-coordinate to center the text
        text_x = int((segment_width * i) + (w - text_size[0]) / 2)

        # Add text caption
        cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)

    final_result = cv2.vconcat([caption_space, combined_results])

    cv2.imshow('Depth Anything', final_result)
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(video_path)
raw_image = ()
while cap.isOpened():
    # Capture frame-by-frame
    ret, raw_image = cap.read()
    
    # Display the resulting frame
    cv2.imshow('Camera', raw_image)
    
    # Wait for a key press and check if it's the 's' key (for 'save')
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("Image captured successfully!")
        break # Break out of the loop to stop capturing

    # Check if the 'q' key is pressed to quit
    if key == ord('q'):
        break

raw_image = cv2.resize(raw_image, (640, 480))
# Save the raw image
cv2.imwrite('raw_image.png', raw_image)
print("Image saved!")

# Process the captured image
depth_estimation(raw_image)

color_raw = o3d.io.read_image("raw_image.png")
depth_raw = o3d.io.read_image("depth_image.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)

# Plot the images
plt.subplot(1, 2, 1)
plt.title('Grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()

# Camera intrinsic parameters built into Open3D for Prime Sense
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# Create the point cloud from images and camera intrinsic parameters
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

# Flip it, otherwise the point cloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# Estimate the bounding box
bbox = pcd.get_oriented_bounding_box()

# Get dimensions of the bounding box
dimensions = bbox.get_extent()

print("Dimensions (length, width, height):", dimensions)