import tensorflow as tf
import cv2
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, ConvNextModel


def load_image_from_blob_cv(blob_img, container_client):
    """
    Loads the image from the Azure Blob Storage using OpenCV and returns it as a numpy array.
    Args:
        blob_img (str): name of the blob image in the container
        container_client (azure.storage.blob.BlobContainerClient): container client
    Returns:
        (numpy.ndarray): loaded image in greyscale
    """
    blob_client = container_client.get_blob_client(blob_img)
    streamdownloader = blob_client.download_blob()
    blob_data = streamdownloader.readall()
    image_array = np.asarray(bytearray(blob_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_image_from_blob_tf(blob_filename, account_name, container_name):
    """
    Loads the image from the Azure Blob Storage using tensorflow and returns it as a numpy array.
    Args:
        blob_img (str): name of the blob image in the container
        account_name (str): Azure Storage Account Name
        container_name (str): Azure Storage Container Name
    Returns:
        (numpy.ndarray): loaded image in greyscale
    """
    full_path = f'az://{account_name}/{container_name}/{blob_filename}'
    with tf.io.gfile.GFile(full_path, 'rb') as f:
        img = tf.image.decode_jpeg(f.read(), channels=3)
    image_array = img.numpy()
    # img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return image_array


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


def crop_image_cv(image, size=(600, 600)):
    h, w = image.shape[:2]
    center_h, center_w = h // 2, w // 2
    start_h = max(center_h - size[0] // 2, 0)
    start_w = max(center_w - size[1] // 2, 0)
    return image[start_h:start_h + size[0], start_w:start_w + size[1]]


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


# # Approximate K, to be updated when NASA sends us actual values
# fx = fy = 1.4 * image_width / (2 * np.tan(np.radians(185) / 2))
# cx = image_width / 2
# cy = image_height / 2
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# # Approximate D, to be updated when NASA sends us actual values
# D = np.array([-0.3, 0.1, 0.0, 0.0])


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


def visualize_augmentations_cv(image_list, account_name=None, container_name=None, container_client=None, enhance=True, correct_fisheye=True, crop=True, use_tf=True):
    """
    Visualizes original and augmented greyscale images using OpenCV.
    Args:
        image_list (list): List of image paths in Azure Blob Storage
        account_name (str): Azure Storage Account Name
        container_name (str): Azure Storage Container Name
        container_client (azure.storage.blob.BlobServiceClient): Azure Blob Storage Client
        enhance (bool): Whether to apply image augmentation
        correct_fisheye (bool): Whether to apply fisheye distortion correction
        crop (bool): Whether to apply image cropping
    Returns:
        None
    """
    fig, axes = plt.subplots(len(image_list), 3, figsize=(15, 5 * len(image_list)))

    for idx, blob_img in enumerate(image_list):
        if use_tf:
            original_image = load_image_from_blob_tf(blob_img, account_name, container_name)
        else:
            original_image = load_image_from_blob_cv(blob_img, container_client)
        greyscale_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        augmented_image = greyscale_image.copy()

        if correct_fisheye:
            augmented_image = undistort_fisheye_image(augmented_image)
        if crop:
            augmented_image = crop_and_correct_image_cv(augmented_image, size1=(1500,1500), offset=(85, 180), size2=(1000,1000))
        if enhance:
            augmented_image = augment_greyscale_image(augmented_image, contrast_factor=1.5, brightness_beta=30, kernel_size=(5, 5))

        axes[idx, 0].imshow(original_image)
        axes[idx, 0].set_title("Original Image (RGB)")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(greyscale_image, cmap='gray')
        axes[idx, 1].set_title("Greyscale Image")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(augmented_image, cmap='gray')
        axes[idx, 2].set_title("Augmented Image")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()


def extract_features(blob_img, image_processor, model, account_name=None, container_name=None, container_client=None, enhance=True, correct_fisheye=False, use_tf=False):
    """
    Extracts feature maps from the model by:
        - loading, cropping and preprocessing the image using OpenCV
        - forward pass through the model
        - extracting the last hidden state to produce feature maps
    Args:
        blob_img (str): name of the blob image in the container
        image_processor (torch.nn.Module): image processor model
        model (torch.nn.Module): loaded ConvNext model to extract feature maps
        account_name (str): Azure Storage Account Name
        container_name (str): Azure Storage Container Name
        container_client (azure.storage.blob.BlobServiceClient): Azure Blob Storage Client
        enhance (bool): whether to apply image augmentation
        correct_fisheye (bool): whether to correct fisheye distortion
        use_tf (bool): whether to use TensorFlow to load the image
    Returns:
        (torch.Tensor): feature maps
    """
    if use_tf:
        image = load_image_from_blob_tf(blob_img, account_name, container_name)
    else:
        image = load_image_from_blob_cv(blob_img, container_client)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if correct_fisheye:
        image = undistort_fisheye_image(image)

    image = crop_and_correct_image_cv(image, size1=(1500, 1500), offset=(85, 180),
                                                    size2=(1000, 1000))

    if enhance:
        image = augment_greyscale_image(image, contrast_factor=1.5, brightness_beta=30, kernel_size=(5, 5))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    inputs = image_processor(image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    feature_map = outputs.last_hidden_state

    return feature_map


def extract_gradients(blob_img, image_processor, model, account_name=None, container_name=None, container_client=None, enhance=True, correct_fisheye=False, use_tf=False):
    """
    Computes gradients from the model by:
        - loading, cropping and preprocessing the image using OpenCV
        - enabling gradient computation
        - forward pass through the model
        - gradient computation
    Args:
        blob_img (str): name of the blob image in the container
        model: loaded model
        account_name (str): Azure Storage Account Name
        container_name (str): Azure Storage Container Name
        container_client (azure.storage.blob.BlobServiceClient): Azure Blob Storage Client
        enhance (bool): whether to apply image augmentation
        correct_fisheye (bool): whether to correct fisheye distortion
        use_tf (bool): whether to use TensorFlow for image loading
    Returns:
        (torch.Tensor): gradients
    """
    if use_tf:
        image = load_image_from_blob_tf(blob_img, account_name, container_name)
    else:
        image = load_image_from_blob_cv(blob_img, container_client)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if correct_fisheye:
        image = undistort_fisheye_image(image)

    image = crop_and_correct_image_cv(image, size1=(1500, 1500), offset=(85, 180),
                                      size2=(1000, 1000))

    if enhance:
        image = augment_greyscale_image(image, contrast_factor=1.5, brightness_beta=30, kernel_size=(5, 5))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    inputs = image_processor(image_rgb, return_tensors="pt")
    inputs['pixel_values'].requires_grad_()

    outputs = model(**inputs)

    grad_feature_map = outputs.last_hidden_state

    grad_outputs = torch.ones_like(grad_feature_map)
    gradients = torch.autograd.grad(
        outputs=grad_feature_map,
        inputs=inputs['pixel_values'],
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]

    return gradients


def apply_regularization(model, weight_decay=1e-4):
    """
    Applies L2 regularization to the model.
    Args:
        model: loaded model
        weight_decay (float): regularization parameter
    Returns:
      None
    """
    for param in model.parameters():
        param.data = param.data - weight_decay * param.data


def visualize_feature_maps(image_list, model_list, account_name=None, container_name=None, container_client=None, enhance=True, correct_fisheye=True, crop=True, overlay_grad=False, use_tf=True):
    """
    Visualizes original images, augmented images, and feature overlays.
    Args:
        image_list (list): List of image paths in Azure Blob Storage
        model_list (list): List of models to extract feature maps
        account_name (str): Azure Storage Account Name
        container_name (str): Azure Storage Container Name
        container_client (azure.storage.blob.BlobServiceClient): Azure Blob Storage Client
        enhance (bool): Whether to apply image augmentation
        correct_fisheye (bool): Whether to correct fisheye distortion
        crop (bool): Whether to apply image cropping
        overlay_grad (bool): Whether to overlay gradients on the images
        use_tf (bool): Whether to use TensorFlow to load images
    Returns:
        None
    """
    fig, axes = plt.subplots(len(image_list), 3 + len(model_list), figsize=(4 * len(image_list), 6 * len(image_list)))

    for idx, blob_img in enumerate(image_list):
        if use_tf:
            original_image = load_image_from_blob_tf(blob_img, account_name, container_name)
        else:
            original_image = load_image_from_blob_cv(blob_img, container_client)
        greyscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        augmented_image = greyscale_image.copy()
        if correct_fisheye:
            augmented_image = undistort_fisheye_image(augmented_image)
        if crop:
            augmented_image = crop_and_correct_image_cv(augmented_image, size1=(1500, 1500), offset=(85, 180),
                                              size2=(1000, 1000))
        if enhance:
            augmented_image = augment_greyscale_image(augmented_image)

        # Display images
        axes[idx, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title("Original Image (RGB)")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(greyscale_image, cmap='gray')
        axes[idx, 1].set_title("Greyscale Image")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(augmented_image, cmap='gray')
        axes[idx, 2].set_title("Augmented Greyscale Image")
        axes[idx, 2].axis('off')

        for i, model_name in enumerate(model_list):
            image_processor = AutoImageProcessor.from_pretrained(model_name)
            model = ConvNextModel.from_pretrained(model_name)
            model.eval()
            # Extract features and gradients
            gradients = extract_gradients(blob_img, image_processor, model, account_name, container_name, container_client, enhance=False, use_tf=use_tf)
            features = extract_features(blob_img, image_processor, model, account_name, container_name, container_client, enhance=False, use_tf=use_tf)

            if overlay_grad and gradients is not None:
                summed_gradients = gradients.squeeze().mean(dim=0).detach().numpy()
                summed_gradients = (summed_gradients - summed_gradients.min()) / (summed_gradients.max() - summed_gradients.min())
                summed_gradients_img = Image.fromarray((summed_gradients * 255).astype(np.uint8))
                size = (augmented_image.shape[1], augmented_image.shape[0])
                gradient_map_image = summed_gradients_img.resize(size, resample=Image.BILINEAR)
                gradient_map_image = np.array(gradient_map_image).astype(np.float32)

                axes[idx, 2 + i + 1].imshow(augmented_image, cmap='gray')
                axes[idx, 2 + i + 1].imshow(gradient_map_image, cmap='gray', alpha=0.5)
                axes[idx, 2 + i + 1].set_title(f'{model_name.split("/")[1]}')
                axes[idx, 2 + i + 1].axis('off')

            else:
                summed_feature_map = features.squeeze().mean(dim=0).detach().numpy()
                summed_feature_map = (summed_feature_map - summed_feature_map.min()) / (summed_feature_map.max() - summed_feature_map.min())
                summed_feature_map_img = Image.fromarray((summed_feature_map * 255).astype(np.uint8))
                size = (augmented_image.shape[1], augmented_image.shape[0])
                feature_map_image = summed_feature_map_img.resize(size, resample=Image.BILINEAR)
                feature_map_image = np.array(feature_map_image).astype(np.float32)

                axes[idx, 2 + i + 1].imshow(augmented_image)
                axes[idx, 2 + i + 1].imshow(feature_map_image, cmap='jet', alpha=0.5)
                axes[idx, 2 + i + 1].set_title(f'{model_name}'.split('/')[1])
                axes[idx, 2 + i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def random_crop_sequence(image_sequence, crop_size_minimum=224):
    """
    Apply identical random crop to a sequence of images and ensure randomness in cropping location.
    The crop size is randomly selected between the minimum size (crop_size_minimum) and the full image size.

    Args:
        image_sequence: A list or tensor of images in the sequence (each image should be the same size).
        crop_size_minimum: The minimum crop size (default is 224).

    Returns:
        cropped_sequence: The cropped sequence of images.
        new_center_coords: The new coordinates of the center pixel after cropping.
    """
    # Get the height and width of the original images
    orig_height, orig_width = image_sequence[0].shape[-2:]

    # Randomly select the crop size between the minimum and full image size
    max_crop_size = min(orig_height, orig_width)
    crop_h = random.randint(crop_size_minimum, max_crop_size)
    crop_w = crop_h  # Keeping the crop square for simplicity

    # Ensure that the crop size is smaller than the original image
    assert crop_h <= orig_height and crop_w <= orig_width, "Crop size must be smaller than the original image size."

    # Original center coordinates of the image
    orig_center_x = orig_width // 2
    orig_center_y = orig_height // 2

    # Randomly select the top-left corner for the crop
    top = random.randint(0, orig_height - crop_h)
    left = random.randint(0, orig_width - crop_w)

    # Apply the same crop to each image in the sequence
    cropped_sequence = [TF.crop(img, top, left, crop_h, crop_w) for img in image_sequence]

    # Calculate the new center coordinates based on the original center relative to the cropped region
    new_center_x = orig_center_x - left  # Adjust the original center x based on the crop left offset
    new_center_y = orig_center_y - top   # Adjust the original center y based on the crop top offset

    # Ensure the new center coordinates are still within the cropped image bounds
    new_center_x = max(0, min(new_center_x, crop_w - 1))
    new_center_y = max(0, min(new_center_y, crop_h - 1))

    return cropped_sequence, (new_center_x, new_center_y)


def resize_sequence_and_adjust_center(cropped_sequence, new_center_coords, target_size=(224, 224)):
    """
    Resize the cropped sequence to a target size and adjust the center coordinates accordingly.

    Args:
        cropped_sequence: The list of cropped images.
        new_center_coords: The (x, y) coordinates of the center in the cropped image.
        target_size: The desired output size (height, width), default is (224, 224).

    Returns:
        resized_sequence: The resized sequence of images.
        resized_center_coords: The adjusted (x, y) coordinates of the center in the resized images.
    """
    crop_h, crop_w = cropped_sequence[0].shape[-2:]  # Get height and width of the cropped image
    target_h, target_w = target_size  # Desired target size (e.g., 224x224)

    # Calculate the scaling factors for height and width
    scale_x = target_w / crop_w
    scale_y = target_h / crop_h

    # Resize each image in the sequence to the target size
    resized_sequence = [TF.resize(img, size=target_size) for img in cropped_sequence]

    # Adjust the center coordinates according to the scaling factor
    new_center_x, new_center_y = new_center_coords
    resized_center_x = int(new_center_x * scale_x)
    resized_center_y = int(new_center_y * scale_y)

    return resized_sequence, (resized_center_x, resized_center_y)