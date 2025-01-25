# Monocular-Vision-for-Depth-Estimation-and-3D-Point-Cloud-Generation

This project demonstrates real-time depth estimation and 3D point cloud generation using the Depth Anything model, OpenCV, and Open3D. It captures frames from a video feed, estimates their depth, and generates 3D point clouds with bounding box analysis.

**Features**

* Real-time depth estimation using the Depth Anything model.
* RGB-D image creation and visualization with Open3D.
* 3D point cloud generation and bounding box dimension calculation.
* Captures images directly from a video stream or webcam.
* Saves and visualizes depth images and point clouds.

**Prerequisites**

Ensure the following dependencies are installed:
* Python 3.8 or higher
* PyTorch
* OpenCV
* Open3D
* Matplotlib
* tqdm

To install the required Python packages, run:
**pip install -r requirements.txt**

Note: The Depth Anything model requires a CUDA-compatible GPU for optimal performance.

**Usage**

Running the Code

1. Clone the repository:
**git clone https://github.com/<your-repo>/depth-anything**
**cd depth-anything**

2.  Download the pretrained Depth Anything model. This will be handled automatically by the script if the model is not already downloaded.

3. Run the script:
**python depth_estimation.py**

4. Use the following controls during execution:
Press s to capture an image and generate the depth map.
Press q to quit the video feed.

**Output Files**

* raw_image.png: The raw image captured from the video feed.
* depth_image.png: The depth map of the captured image.
* output_point_cloud.ply: The 3D point cloud file generated from the RGB-D image.

**Key Components**

Depth Estimation:

The depth estimation process uses the pretrained Depth Anything model with the following steps:
* Transform the input image (resize, normalize, etc.).
* Generate a depth map using the model.
* Normalize and apply a colormap for visualization.

RGB-D and 3D Point Cloud:
* An RGB-D image is created using the raw and depth images.
* The point cloud is generated using the RGB-D image and intrinsic camera parameters.
* The point cloud is saved in .ply format and visualized in Open3D.

Bounding Box Dimensions:
* The script calculates the oriented bounding box of the point cloud and outputs its dimensions (length, width, and height).

**Customization**

Encoder Type: Change the encoder variable to use different model encoders (vits, vitb, vitl).
Camera Source: Modify the video_path variable to set a custom video source. Use 0 for the default webcam.
Image Size: Adjust the Resize parameters in the transform pipeline.

**Dependencies Installation**

Ensure your environment supports CUDA for GPU acceleration. Install the required libraries:
**pip install torch torchvision opencv-python open3d matplotlib tqdm**

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Contributions**

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.
