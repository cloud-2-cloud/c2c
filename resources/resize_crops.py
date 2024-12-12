import os
import time
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from azure.storage.blob import BlobServiceClient
import numpy as np
import cv2


def load_image_from_blob_cv(blob_img, container_client):
    """
    Loads the image from Azure Blob Storage using OpenCV and returns it as a numpy array.
    Args:
        blob_img (str): name of the blob image in the container
        container_client (azure.storage.blob.BlobContainerClient): container client
    Returns:
        (numpy.ndarray): loaded image with original channels retained
    """
    blob_client = container_client.get_blob_client(blob_img)
    streamdownloader = blob_client.download_blob()
    blob_data = streamdownloader.readall()
    image_array = np.asarray(bytearray(blob_data), dtype=np.uint8)
    # img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def undistort_fisheye_image(distorted_image):
    """
    Apply correction for fisheye distortion.
    Args:
        distorted_image (numpy.ndarray): original image that has the fisheye distortion.
     Returns:
        (numpy.ndarray): undistorted image with fisheye correction.
    """
    # Parameters provided
    f = 1.4  # focal length [mm]
    mu = 2.8e-3  # pixel pitch [mm]
    S = 2  # output (undistorted) image scale factor
    # distortion polynomial order:  [2 4 6 8]
    # polynomial coefficients:
    coeffs = np.array([0.01166363, -0.04819808, 0.07918044, -0.037572])

    H = distorted_image.shape[0]  # image height [pixel]
    W = distorted_image.shape[1]  # image width [pixel]
    cx = (W - 1) / 2  # image center coordinate [pixel]
    cy = (H - 1) / 2  # image center coordinate [pixel]

    K = np.array([[f / mu, 0, cx], [0, f / mu, cy], [0, 0, 1]])

    # compute intrinsic matrix for undistorted image
    cpx = (W * S - 1) / 2
    cpy = (H * S - 1) / 2
    P = np.array([[f / mu, 0, cpx], [0, f / mu, cpy], [0, 0, 1]])

    # rectification matrix (identity)
    R = np.eye(3)

    # produce undistorted image
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K=K, D=coeffs, R=R, P=P, size=[W * S, H * S], m1type=cv2.CV_16SC2)
    undistorted_image = cv2.remap(distorted_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return undistorted_image


def crop_and_correct_image_cv(image, size1=(1500, 1500), offset=(85, 180), size2=(1000, 1000)):
    h, w = image.shape[:2]

    # Cropping it to 1500X1500 and accounting for offset.
    center_h, center_w = h // 2, w // 2
    offset_h, offset_w = offset
    start_h = min(max(center_h - size1[0] // 2 + offset_h, 0), h - size1[0])
    start_w = min(max(center_w - size1[1] // 2 + offset_w, 0), w - size1[1])
    cropped_image = image[start_h:start_h + size1[0], start_w:start_w + size1[1]]
    h, w = cropped_image.shape[:2]

    # further crop the image to 1000X1000
    center_h, center_w = h // 2, w // 2
    start_h = max(center_h - size2[0] // 2, 0)
    start_w = max(center_w - size2[1] // 2, 0)

    return cropped_image[start_h:start_h + size2[0], start_w:start_w + size2[1]]


def crop_and_correct_image_cv2(image, size1=(1500, 1500), offset=(85, 180), size2=(800, 800)):
    """
    Crops the undistorted image in two stages and ensures the original center (960, 540)
    aligns with the center of the final cropped image (400, 400).

    Args:
        image (numpy.ndarray): Input image to crop.
        size1 (tuple): Size of the first crop (width, height).
        offset (tuple): Offset for the first crop.
        size2 (tuple): Size of the final crop (width, height).

    Returns:
        numpy.ndarray: Cropped image.
    """
    h, w = image.shape[:2]

    # Step 1: First crop with offset
    center_h, center_w = h // 2, w // 2  # Center of the undistorted image
    offset_h, offset_w = offset

    # Adjust the starting coordinates for the first crop
    start_h1 = max(center_h - size1[1] // 2 + offset_h, 0)
    start_w1 = max(center_w - size1[0] // 2 + offset_w, 0)
    cropped_image = image[start_h1:start_h1 + size1[1], start_w1:start_w1 + size1[0]]

    # Step 2: Adjust second crop to ensure the original center aligns with the center of the crop-corrected image
    crop_h, crop_w = size2
    center_h_crop = center_h - start_h1  # Adjusted center in the cropped image
    center_w_crop = center_w - start_w1

    # Calculate start coordinates to place the center at 400, 400
    start_h2 = max(center_h_crop - crop_h // 2, 0)
    start_w2 = max(center_w_crop - crop_w // 2, 0)

    final_image = cropped_image[start_h2:start_h2 + crop_h, start_w2:start_w2 + crop_w]

    return final_image


def augment_greyscale_image(img, contrast_factor=1.5, brightness_beta=30, kernel_size=(5, 5), blur=True):
    """
    Augments the Greyscale image to enhance the feature of the cloud. Includes:
      1. Contrast and Brightness
      2. Gaussian Blur
      3. Histogram Equalization
      4. Sharpening
    Args:
        img (numpy.ndarray): greyscale image
        contrast_factor (float): contrast factor
        brightness_beta (int): brightness factor
        kernel_size (tuple): kernel size for Gaussian Blur
        blur (bool): whether to apply Gaussian Blur
    Returns:
        (numpy.ndarray): augmented image
    """
    img_enhanced = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_beta)
    if blur:
        img_enhanced = cv2.GaussianBlur(img_enhanced, kernel_size, 0)
    img_enhanced = cv2.equalizeHist(img_enhanced)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_enhanced)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img_enhanced, -1, kernel)


def save_cropped_image(container_client, cropped_img, src_blob_name, dest):
    """
    Saves the cropped image to Azure Blob Storage with a modified folder name using OpenCV.

    Args:
        container_client (azure.storage.blob.ContainerClient): Azure blob storage container client
        cropped_img (numpy.ndarray): Cropped image as a numpy array.
        src_blob_name (str): Source blob name.
        dest (str): The folder where cropped images will be saved.

    Returns:
        None
    """
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    is_success, buffer = cv2.imencode('.jpg', cropped_img_rgb)
    if not is_success:
        raise ValueError("Failed to encode image with OpenCV.")

    img_byte_array = BytesIO(buffer)

    base_name = os.path.basename(src_blob_name)
    cropped_blob_name = os.path.join(dest, base_name)

    blob_client = container_client.get_blob_client(cropped_blob_name)
    blob_client.upload_blob(img_byte_array.getvalue(), overwrite=True)


def crop_and_correct_images_in_blob(container_client, src, dest, print_cnts=False):
    """
    Reads images from src dir in an azure blob, applies crop_to_square,
    and uploads cropped images to dest dir in the same blob.

    Args:
        container_client (azure.storage.blob.ContainerClient): azure blob storage container client
        src (str): path to the dir containing src images.
        dest (str): path to the dir with cropped images.
        print_cnts (bool): whether to print the number of images processed.

    Returns:
        (str): path to the cropped images directory.
    """
    blobs = container_client.list_blobs(name_starts_with=src)
    existing_blobs = {blob.name for blob in container_client.list_blobs(name_starts_with=dest)}
    if print_cnts:
        print(len([blob.name for blob in container_client.list_blobs(name_starts_with=src)]))
        print(len(existing_blobs))
    for blob in blobs:
        dir = os.path.dirname(blob.name)
        if dir.endswith('_frames') and blob.name.endswith('.jpg'):
            base_name = os.path.basename(blob.name)
            new_filename = base_name.replace('.jpg', '_crop_corrected.jpg')
            cropped_blob_name = os.path.join(dest, new_filename)
            if cropped_blob_name in existing_blobs:
                continue
            img_stream = load_image_from_blob_cv(blob.name, container_client)
            fisheye_corrected_img = undistort_fisheye_image(img_stream)
            cropped_img = crop_and_correct_image_cv2(fisheye_corrected_img)
            save_cropped_image(container_client, cropped_img, blob.name, dest)
    return dest


def draw_crop_box_on_image(image, size1=(1500, 1500), offset=(85, 180), size2=(1000, 1000)):
    """
    Draws a red bounding box on the image to indicate the crop area.

    Args:
        image (numpy.ndarray): The undistorted image on which to draw the crop box.
        size1 (tuple): Size of the initial crop (width, height).
        offset (tuple): Offset applied to the initial crop.
        size2 (tuple): Final crop size after the first crop.

    Returns:
        numpy.ndarray: Image with the red bounding box indicating the crop area.
    """
    h, w = image.shape[:2]

    # Calculate the coordinates for the first crop with offset
    center_h, center_w = h // 2, w // 2
    offset_h, offset_w = offset
    start_h = min(max(center_h - size1[0] // 2 + offset_h, 0), h - size1[0])
    start_w = min(max(center_w - size1[1] // 2 + offset_w, 0), w - size1[1])

    # Calculate coordinates for the second crop within the initial cropped area
    final_center_h = start_h + size1[0] // 2
    final_center_w = start_w + size1[1] // 2
    final_start_h = max(final_center_h - size2[0] // 2, 0)
    final_start_w = max(final_center_w - size2[1] // 2, 0)

    # Define the bounding box for visualization (on the original image)
    end_h = final_start_h + size2[0]
    end_w = final_start_w + size2[1]

    # Draw rectangle (red bounding box) on the undistorted image
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (final_start_w, final_start_h), (end_w, end_h), (255, 0, 0), 10)

    return image_with_box


def draw_crop_box_on_image2(image, size1=(1500, 1500), offset=(85, 180), size2=(800, 800)):
    """
    Draws a red bounding box on the image to indicate the crop areas, matching the logic of crop_and_correct_image_cv2.

    Args:
        image (numpy.ndarray): The undistorted image on which to draw the crop box.
        size1 (tuple): Size of the first crop (width, height).
        offset (tuple): Offset applied to the initial crop.
        size2 (tuple): Final crop size after the first crop.

    Returns:
        numpy.ndarray: Image with the red bounding box indicating the crop areas.
    """
    h, w = image.shape[:2]

    # Step 1: Calculate the coordinates for the first crop with offset
    center_h, center_w = h // 2, w // 2
    offset_h, offset_w = offset

    # Adjust the starting coordinates for the first crop
    start_h1 = max(center_h - size1[1] // 2 + offset_h, 0)
    start_w1 = max(center_w - size1[0] // 2 + offset_w, 0)
    end_h1 = start_h1 + size1[1]
    end_w1 = start_w1 + size1[0]

    # Step 2: Calculate the coordinates for the second crop
    crop_h, crop_w = size2
    center_h_crop = center_h - start_h1  # Adjusted center in the cropped image
    center_w_crop = center_w - start_w1

    # Calculate start coordinates to place the center at the target (e.g., 400, 400 in the final cropped image)
    start_h2 = max(center_h_crop - crop_h // 2, 0)
    start_w2 = max(center_w_crop - crop_w // 2, 0)
    end_h2 = start_h2 + crop_h
    end_w2 = start_w2 + crop_w

    # Define the bounding boxes for visualization
    image_with_box = image.copy()

    # # Draw rectangle for the first crop
    # cv2.rectangle(image_with_box, (start_w1, start_h1), (end_w1, end_h1), (0, 255, 0), 5)

    # Draw rectangle for the second crop within the first crop
    cv2.rectangle(image_with_box, (start_w1 + start_w2, start_h1 + start_h2),
                  (start_w1 + end_w2, start_h1 + end_h2), (255, 0, 0), 5)

    return image_with_box
