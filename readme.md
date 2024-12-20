cloud2cloud 
===========
## Cloud-Top Height Field Estimation from Aerial Imagery
### In collaboration with NASA

A Thesis in the Field of Data Science for the Degree of Master of Liberal Arts in Extension Studies

This project aims to address a critical issue in extreme weather forecasting and guarantee accurate measurements of the earth's atmosphere. We are developing predictive models for estimating cloud top heights using advanced deep learning, computer vision, and optical flow geometry techniques. Our objectives are to accurately predict cloud top heights and create height field projections of cloud formations. We utilize high-definition images from NASA's FEGS ER2 mission (flying at 20,000 meters), complemented by LiDAR and aircraft metadata. This involves analyzing cloud motion between frames to determine relative heights through parallax effects, and for some methods using the periodic LiDAR measurements to calibrate these relative heights into absolute measurements. The project explores several computer vision approaches including optical flow estimation, monocular depth prediction, and deep learning techniques.

### Due to this project being heavily research-oriented, the majority of the files are Jupyter notebooks, with some resource and utility Python files.
---
![](img/cloud.jpg)
---
[Full report](docs/cloud2cloud.pdf)
---
# Fully Documented Jupyter notebooks

## RAFT
* [Final RAFT height calculator stitching](https://github.com/cloud-2-cloud/c2c/blob/main/RAFT/final_RAFT_height_calculator_stitching.ipynb)
  - RAFT fine-tuning, height field generation, and stitching with images. Too large to display in GitHub without downloading.
* [Final RAFT height calculator stitching no output](https://github.com/cloud-2-cloud/c2c/blob/main/RAFT/final_RAFT_height_calculator_stitching_no_output.ipynb)
  - RAFT fine-tuning, height field generation, and stitching with no output. Can be viewed in GitHub without downloading the notebook.
* [RAFT appendix](https://github.com/cloud-2-cloud/c2c/blob/main/RAFT/raft_appendix.pdf)
  - The technical information from the RAFT notebook in LaTeX format.

## CNN RNN
* [Final CNN RNN LiDAR point](https://github.com/cloud-2-cloud/c2c/blob/main/cnn_rnn/final_CNN_RNN_LiDAR_point.ipynb)

# Notebooks

## Lucas-Kanade Optical Flow
* [EDA Conv Next Gradient Deep Dive](https://github.com/cloud-2-cloud/c2c/blob/main/Lucas_Kanade_Optical_Flow/EDA_Conv_Next_Gradient_Deep_Dive.ipynb)
  - Initial exploration of out-of-the-box conv_next output for potential use for optical flow geometric model. This was not deemed appropriate for selecting good features to track for Lucas-Kanade but rather an output to be ingested by the deep learning model technique.
* [Initial Image Pixel EDA And Corner Detection](https://github.com/cloud-2-cloud/c2c/blob/main/Lucas_Kanade_Optical_Flow/Initial_Image_Pixel_EDA_And_Corner_Detection.ipynb)
  - Exploratory Data Analysis on cloud pixel luminosity and eventually looking at finding an appropriate threshold to use for Shi-Tomasi. This was all for trying to think of suitable ways to identify "corner" pixels or good cloud pixels in general that can be tracked by Lucas-Kanade.
* [LK Tracking Fisheye Corrections And Center Predictions](https://github.com/cloud-2-cloud/c2c/blob/main/Lucas_Kanade_Optical_Flow/LK_Tracking_Fisheye_Corrections_And_Center_Predictions.ipynb)
  - This notebook has 2 parts. The first part is the visual aid to check that fish-eye correction is working as expected for Lucas-Kanade. It outputs tracking on original image frames and then corrected image frames.
  - The second part is where we read in the test video dataset and output the heights of the center 48x48 pixels that are trackable "corners."

## MiDAS
* [MiDaS](https://github.com/cloud-2-cloud/c2c/blob/main/MiDAS/MiDaS.ipynb)
  - The MiDAS notebook where we tested MiDAS on the cloud imagery, and found it wasn't able to create appropriate height fields. This notebook still contains the images and isn't viewable in GitHub on the web.
* [MiDaS no output](https://github.com/cloud-2-cloud/c2c/blob/main/MiDAS/MiDaS_no_output.ipynb)
  - The MiDAS notebook where we tested MiDAS on the cloud imagery, and found it wasn't able to create appropriate height fields. This notebook has had the output cleared and is viewable in GitHub on the web.

## TV-L1 Optical Flow
* [Center Points TV L1](https://github.com/cloud-2-cloud/c2c/blob/main/TV_L1_Optical_Flow/Center_Points_TV_L1.csv)
  - This CSV is the output for the notebook "Get Height of Center Points 1." This contains the predicted heights for the test dataset video.
* [Create 3D Mesh v2](https://github.com/cloud-2-cloud/c2c/blob/main/TV_L1_Optical_Flow/Create_3D_Mesh_v2.ipynb)
  - This notebook shows heights calculated by stitching height fields for frames from 17:57:04 to 18:09:35. The generated figures show heights at the center line along the aircraft route. In other words, we can generate a 3D mesh of the cloud with viewpoints along the aircraft route and passing through the center of the frames.
* [Get Height For Entire Video v1](https://github.com/cloud-2-cloud/c2c/blob/main/TV_L1_Optical_Flow/Get_Height_For_Entire_Video_v1.ipynb)
  - This notebook calculates heights for all pixels in selected frames using the Nvidia GPU version of TV-L1 from the OpenCV CUDA library. This is used to generate 3-D height fields.
* [Get Height of Center Points 1](https://github.com/cloud-2-cloud/c2c/blob/main/TV_L1_Optical_Flow/Get_Height_of_Center_Points_1.ipynb)
  - This notebook is used to calculate just the center-most pixel's height per each selected frame. Its output is "Center_Points_TV_L1.csv" referenced above. It is slightly optimized runtime-wise compared to the notebook that generated full 3D height fields.
* [TV L1 Visualization](https://github.com/cloud-2-cloud/c2c/blob/main/TV_L1_Optical_Flow/TV_L1_Visualization.ipynb)
  - This notebook visualizes some of the 3D heights generated by the TV-L1 methodology.

## CNN RNN
* [Conv-next-feature-extraction (HTML)](https://github.com/cloud-2-cloud/c2c/blob/main/cnn_rnn/conv-next-feature-extraction.html)
  - Initial exploration of using ConvNext and image augmentation in HTML format.
* [Conv-next-feature-extraction (Notebook)](https://github.com/cloud-2-cloud/c2c/blob/main/cnn_rnn/conv-next-feature-extraction.ipynb)
  - Initial exploration of using ConvNext and image augmentation in Jupyter notebook format.

# Data Processing
* [Aircraft metadata processor](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/aircraft_metadata_processor.ipynb)
* [Cloud2cloud preprocessor](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/cloud2cloud_preprocessor.ipynb)
* [Fisheye](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/fisheye.ipynb)
* [LiDAR processor](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/lidar_processor.ipynb)
* [Resize crops](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/resize_crops.ipynb)
* [Video processor](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/video_processor.ipynb)
* [Video processor TF In Memory](https://github.com/cloud-2-cloud/c2c/blob/main/data_processing/video_processor_TF_In_Memory.ipynb)

# Resources
* [Cloud2cloud ConvNext](https://github.com/cloud-2-cloud/c2c/blob/main/resources/cloud2cloud_ConvNext.py)
* [Cloud2cloud data class](https://github.com/cloud-2-cloud/c2c/blob/main/resources/cloud2cloud_data_class.py)
* [Resize crops](https://github.com/cloud-2-cloud/c2c/blob/main/resources/resize_crops.py)

---

File Organization
------------
```
.
├── Lucas_Kanade_Optical_Flow
│             ├── EDA_Conv_Next_Gradient_Deep_Dive.ipynb
│             ├── Initial_Image_Pixel_EDA_And_Corner_Detection.ipynb
│             └── LK_Tracking_Fisheye_Corrections_And_Center_Predictions.ipynb
├── MiDAS
│             ├── MiDaS.ipynb
│             └── MiDaS_no_output.ipynb
├── RAFT
│             ├── final_RAFT_height_calculator_stitching.ipynb
│             ├── final_RAFT_height_calculator_stitching_no_output.ipynb
│             └── raft_appendix.pdf
├── TV_L1_Optical_Flow
│             ├── Center_Points_TV_L1.csv
│             ├── Create_3D_Mesh_v2.ipynb
│             ├── Get_Height_For_Entire_Video_v1.ipynb
│             ├── Get_Height_of_Center_Points_1.ipynb
│             └── TV_L1_Visualization.ipynb
├── cnn_rnn
│             ├── conv-next-feature-extraction.html
│             ├── conv-next-feature-extraction.ipynb
│             └── final_CNN_RNN_LiDAR_point.ipynb
├── data_processing
│             ├── aircraft_metadata_processor.ipynb
│             ├── cloud2cloud_preprocessor.ipynb
│             ├── fisheye.ipynb
│             ├── lidar_processor.ipynb
│             ├── resize_crops.ipynb
│             ├── video_processor.ipynb
│             └── video_processor_TF_In_Memory.ipynb
└── resources
    ├── cloud2cloud_ConvNext.py
    ├── cloud2cloud_data_class.py
    └── resize_crops.py
```
