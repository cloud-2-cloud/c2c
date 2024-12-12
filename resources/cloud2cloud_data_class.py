#!/usr/bin/env python
# coding: utf-8

import cv2
import io
import numpy as np
import pandas as pd
import os
import torch
import torchvision.transforms.functional as TF
from datetime import datetime
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cloud2cloud_ConvNext as c2c_ConvNext


aircraft_metadata_params = ['DateTime_UTC', 'GPS_MSL_Alt', 'Drift', 'Pitch', 'Roll', 'Vert_Velocity']
CTH_col = 'top_height'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Aircraft Metadata
def load_metadata(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    streamdownloader = blob_client.download_blob()
    metadata_df = pd.read_csv(io.BytesIO(streamdownloader.readall()))
    return metadata_df


# LiDAR Validation Heights
def load_validation_heights(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    streamdownloader = blob_client.download_blob()
    validation_df = pd.read_csv(io.BytesIO(streamdownloader.readall()))
    return validation_df


# Cloud2Cloud Dataset classes integrating all 3 data sources: FEGS Images, Aircraft Metadata and LiDAR Validation Hieghts with temporal alignment
class CloudDataset(Dataset):
    def __init__(self, date_folders, transform=None, normalization_params=None, augmentations=None, apply_normalization=True, apply_crop_and_scale=True):
        self.date_folders = date_folders
        self.transform = transform
        self.data_df = self._prepare_dataframe()
        self.transform = transform
        self.augmentations = augmentations
        self.apply_normalization = apply_normalization
        self.apply_crop_and_scale = apply_crop_and_scale

        # Columns to normalize
        self.columns_to_normalize = ['validation_height', 'GPS_MSL_Alt', 'Drift', 'Pitch', 'Roll', 'Vert_Velocity']

        # Calculate or use provided normalization parameters
        if normalization_params is None and self.apply_normalization:
            self.normalization_params = self._calculate_normalization_params(dataframe, self.columns_to_normalize)
        else:
            self.normalization_params = normalization_params

        # Only normalize if the flag is set
        if self.apply_normalization:
            for col in self.columns_to_normalize:
                col_min = self.normalization_params[col]['min']
                col_max = self.normalization_params[col]['max']
                self.data_df[col] = (self.data_df[col] - col_min) / (col_max - col_min)

        # Track the indices where sequences start
        self.sequence_indices = self._generate_sequence_indices()
        self.len_sequence_indices = len(self.sequence_indices)

    def _prepare_dataframe(self):
        """
        Iterates over the date folders in azure blob storage and loads:
          1. .jpg Images from each sub-directory in the folder with '_tight_crop' in the name.
          2. Aircraft Metadata with 1-1 time alignment with the images.
          3. LiDAR Validation Heights, mapped using timestamp, if not available filled with NaN.
        Creates a df with following columns:
            timestamp, image_path, [...aircraft_metadata_params...], validation_height
        """
        image_paths, timestamps, metadata_rows, validation_heights = [], [], [], []

        for folder in self.date_folders:
            print(f"Processing folder: {folder}")
            folder_image_paths, folder_timestamps, folder_metadata_rows, folder_validation_heights = [], [], [], []

            blob_list = container_client.list_blobs(name_starts_with=folder)
            metadata_path, validation_path = None, None
            for blob in blob_list:
                # extract image paths of all .jpg images in cropped folders
                if blob.name.endswith(".jpg") and "_tight_crop" in blob.name:
                    folder_image_paths.append(blob.name)
                    folder_timestamps.append(self._extract_timestamp_from_filename(blob.name))
                # extract the aircraft metadata file path
                if blob.name.startswith(f"{folder}/IWG1.") and "processed" in blob.name:
                    metadata_path = blob.name
                # extract the LiDAR validation file path
                if blob.name.startswith(f"{folder}/goesrplt_CPL_layers_") and blob.name.endswith("_processed.txt"):
                    validation_path = blob.name

            # load aircraft metadata and LiDAR validation data
            if metadata_path:
                metadata_df = load_metadata(metadata_path)
            if validation_path:
                validation_df = load_validation_heights(validation_path)

            # prepare LiDAR validation data
            validation_df['datetime_combined'] = validation_df['date'] + ' ' + validation_df['timestamp']
            validation_df['datetime_combined'] = validation_df['datetime_combined'].str.split('.').str[0]
            validation_df['datetime_combined'] = pd.to_datetime(validation_df['datetime_combined'], format="%Y-%m-%d %H:%M:%S")

            # prepare aircraft metadata
            metadata_df = metadata_df[aircraft_metadata_params]
            metadata_df['DateTime_UTC'] = metadata_df['DateTime_UTC'].str.split('.').str[0]
            metadata_timestamps = pd.to_datetime(metadata_df['DateTime_UTC'], format="%Y-%m-%d %H:%M:%S")
            metadata_df = metadata_df.set_index(metadata_timestamps)
            aligned_metadata = pd.DataFrame(index=pd.to_datetime(folder_timestamps, format="%H:%M:%S"))
            aligned_metadata = aligned_metadata.join(metadata_df, how='left')
            aligned_metadata = aligned_metadata[aircraft_metadata_params]

            folder_metadata_rows.extend(aligned_metadata.values.tolist())

            # extract LiDAR validation height exactly matching the timestamp where available, else NaN
            for ts in folder_timestamps:
                cth = self._map_timestamp_to_lidar(ts, validation_df)
                folder_validation_heights.append(cth)

            # Create a folder-level DataFrame
            folder_data = {
                'timestamp': folder_timestamps,
                'image_path': folder_image_paths,
                **{param: [row[i] for row in folder_metadata_rows] for i, param in enumerate(aircraft_metadata_params)},
                'validation_height': folder_validation_heights
            }
            folder_df = pd.DataFrame(folder_data)

            # Remove rows after the last valid validation_height in this folder
            last_valid_index = folder_df['validation_height'].last_valid_index()
            if last_valid_index is not None:
                folder_df_cleaned = folder_df.loc[:last_valid_index].copy()  # Use .copy() to ensure independence
            else:
                folder_df_cleaned = folder_df.copy()  # In case there are no valid entries

            # Extend to the global lists
            image_paths.extend(folder_df_cleaned['image_path'].tolist())
            timestamps.extend(folder_df_cleaned['timestamp'].tolist())
            metadata_rows.extend(folder_df_cleaned[aircraft_metadata_params].values.tolist())
            validation_heights.extend(folder_df_cleaned['validation_height'].tolist())

            # Print the lengths for the current folder
            print(f"Folder {folder}:")
            print(f"  Number of images: {len(folder_df_cleaned['image_path'])}")
            print(f"  Number of timestamps: {len(folder_df_cleaned['timestamp'])}")
            print(f"  Number of metadata rows: {len(folder_df_cleaned)}")
            print(f"  Number of validation heights: {len(folder_df_cleaned['validation_height'])}")

        # Print the final lengths after processing all folders
        print("After processing all folders combined:")
        print(f"  Total number of images: {len(image_paths)}")
        print(f"  Total number of timestamps: {len(timestamps)}")
        print(f"  Total number of metadata rows: {len(metadata_rows)}")
        print(f"  Total number of validation heights: {len(validation_heights)}")

        # Check for any mismatches
        if not (len(image_paths) == len(timestamps) == len(metadata_rows) == len(validation_heights)):
            print("Error: Length mismatch detected!")
            print(f"  Images: {len(image_paths)}")
            print(f"  Timestamps: {len(timestamps)}")
            print(f"  Metadata rows: {len(metadata_rows)}")
            print(f"  Validation heights: {len(validation_heights)}")
            return None

        # combine all aligned data in a df
        data = {
            'timestamp': timestamps,
            'image_path': image_paths,
            **{param: [row[i] for row in metadata_rows] for i, param in enumerate(aircraft_metadata_params)},
            'validation_height': validation_heights
        }
        df = pd.DataFrame(data)
        df = df.drop(columns=['DateTime_UTC'])

        # Add sequence length information for RNN
        self._add_sequence_length_column(df)
        return df

    def _extract_timestamp_from_filename(self, filename):
        """
        Extracts the timestamp from the image filename on the blob.
        path/to/blob/YYYYMMDD_HHMMSS_frame_n_cropped.jpg -> %Y%m%d%H%M%S
        """
        filename = os.path.basename(filename)
        date_str = filename.split("_")[0]
        time_str = filename.split("_")[1]
        timestamp = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return timestamp

    def _map_timestamp_to_lidar(self, timestamp, validation_df):
        """
        extract LiDAR validation height exactly matching the timestamp where available, else NaN
        """
        validation_df['datetime_combined'] = pd.to_datetime(validation_df['datetime_combined'], format="%Y-%m-%d %H:%M:%S")
        timestamp_dt = pd.to_datetime(timestamp, format="%Y-%m-%d %H:%M:%S")
        exact_match = validation_df[validation_df['datetime_combined'] == timestamp_dt]
        return exact_match[CTH_col].values[0] if not exact_match.empty else np.nan

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Retrieve a data record from the dataset for a given index.
        Returns loaded and transformed image, metadata and validation height associated with that image if available.
        """
        row = self.data_df.iloc[idx]

        image_path = row['image_path']
        blob_client = container_client.get_blob_client(image_path)
        streamdownloader = blob_client.download_blob()
        img_data = streamdownloader.readall()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        metadata = torch.tensor([row[param] for param in aircraft_metadata_params], dtype=torch.float32)
        validation_height = torch.tensor([row['validation_height']], dtype=torch.float32)

        return img, metadata, validation_height

    def _add_sequence_length_column(self, df):
        # Initialize a new column to NaN
        df['sequence_length'] = np.nan

        # Track the start of each sequence
        sequence_start = 0

        # Iterate through the dataframe to detect when a validation_height exists
        for i in range(len(df)):
            if not pd.isna(df.loc[i, 'validation_height']):
                # We found the end of a sequence, so mark the previous sequence images
                sequence_length = i - sequence_start
                df.loc[sequence_start:i, 'sequence_length'] = sequence_length + 1  # Using count (1-based index)
                sequence_start = i + 1  # Move the start to the next sequence

        # Ensure all sequence_length values are integers (if any were missed)
        df['sequence_length'] = df['sequence_length'].astype(int)
        return df

    def _calculate_normalization_params(self, dataframe, columns_to_normalize):
        """
        Manually calculate min and max values for the columns and store them for consistency across datasets.
        """
        params = {}
        for col in columns_to_normalize:
            params[col] = {
                'min': dataframe[col].min(),
                'max': dataframe[col].max()
            }
        return params

    def denormalize_validation_height(self, normalized_height):
        """
        Denormalize the validation height using the stored min and max values for validation_height.

        Args:
            normalized_height (float or np.array): The normalized validation height(s) to denormalize.

        Returns:
            float or np.array: The denormalized validation height(s).
        """
        # Get min and max for validation_height
        col_min = self.normalization_params['validation_height']['min']
        col_max = self.normalization_params['validation_height']['max']

        print(f"  Max: {col_max} Min: {col_min}")

        # Denormalize using the stored min and max values
        original_height = normalized_height * (col_max - col_min) + col_min
        return original_height

    def denormalize_flight_data(self, normalized_flight_data):
        """
        Denormalize flight data using the stored min and max values for each column.

        Args:
            normalized_flight_data (np.array): Normalized flight data array to denormalize.

        Returns:
            np.array: Denormalized flight data.
        """
        denormalized_flight_data = []
        for i, col in enumerate(self.columns_to_normalize[1:]):  # Skip validation_height
            col_min = self.normalization_params[col]['min']
            col_max = self.normalization_params[col]['max']
            denorm_value = normalized_flight_data[i] * (col_max - col_min) + col_min
            denormalized_flight_data.append(denorm_value)
        return np.array(denormalized_flight_data)

    def _generate_sequence_indices(self):
        """
        Generate a list of indices where each sequence starts.
        If a NaN is encountered in the sequence_length, the process stops.
        """
        sequence_indices = []
        idx = 0

        while idx < len(self.data_df):
            sequence_length = self.data_df.iloc[idx]['sequence_length']

            if pd.isna(sequence_length):
                # Stop if sequence_length is NaN (no more sequences)
                break

            # Convert sequence_length to an integer
            sequence_length = int(sequence_length)
            sequence_indices.append(idx)
            idx += sequence_length  # Move to the start of the next sequence

        return sequence_indices

    def fetch_item(self, idx):
        # Get the starting index of the sequence
        start_idx = self.sequence_indices[idx]

        # Get the sequence length for this specific starting point
        sequence_length = int(self.data_df.iloc[start_idx]['sequence_length'])

        # Fetch the image sequence from Azure Blob
        image_sequence = []

        # Distortion coefficients (D) - estimated
        D = np.array([-0.34, 0.12, -0.01, 0.0], dtype=np.float64)

        for i in range(sequence_length):
            image_path = self.data_df.iloc[start_idx + i]['image_path']

            # Load image using OpenCV
            img_rgb = c2c_ConvNext.load_image_from_blob_cv(image_path)

            # Get the height and width of the loaded image
            h, w = img_rgb.shape[:2]

            # Dynamically adjust the camera matrix (K) based on the image dimensions
            K = np.array([[w, 0, w / 2],
                          [0, h, h / 2],
                          [0, 0, 1]], dtype=np.float64)

            # Undistort the fisheye image
            # img_rgb = undistort_fisheye_image(img_rgb, K, D, balance=0.5)

            # Convert RGB to grayscale
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            # Apply augmentations to grayscale image
            img_gray = c2c_ConvNext.augment_greyscale_image(img_gray)

            # Convert grayscale image to RGB format
            img_rgb_converted = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

            # Apply additional augmentations to the RGB image if any
            if self.augmentations:
                img_rgb_converted = self.augmentations(img_rgb_converted)

            # Convert back to tensor and append to the sequence
            img_tensor = TF.to_tensor(img_rgb_converted)
            image_sequence.append(img_tensor)

        # # Apply random cropping to the entire sequence with varying crop sizes (minimum of 224, up to full image size)
        # cropped_sequence, new_center_coords = random_crop_sequence(image_sequence, crop_size_minimum=224)

        # # Resize the cropped sequence to 224x224 and adjust the center coordinates accordingly
        # resized_sequence, resized_center_coords = resize_sequence_and_adjust_center(cropped_sequence, new_center_coords)

        # # Stack images into a tensor
        # image_sequence = torch.stack(resized_sequence)

        # Apply random cropping and resizing only if `apply_crop_and_scale` is True
        if self.apply_crop_and_scale:
            # Apply random cropping to the entire sequence with varying crop sizes (minimum of 224, up to full image size)
            cropped_sequence, new_center_coords = c2c_ConvNext.random_crop_sequence(image_sequence, crop_size_minimum=224)

            # Resize the cropped sequence to 224x224 and adjust the center coordinates accordingly
            resized_sequence, resized_center_coords = c2c_ConvNext.resize_sequence_and_adjust_center(cropped_sequence,
                                                                                        new_center_coords)

            # Stack images into a tensor
            image_sequence = torch.stack(resized_sequence)
        else:
            # Resize original images (whatever their original size) to 224x224
            resized_sequence = [TF.resize(img, size=(224, 224)) for img in image_sequence]

            # Stack the resized images
            image_sequence = torch.stack(resized_sequence)

            # Since no cropping is applied, calculate the center of the resized 224x224 image
            resized_center_coords = (112, 112)  # Center of a 224x224 image is (112, 112)

        # Fetch the additional flight data from the dataframe
        flight_data = self.data_df.iloc[start_idx:start_idx + sequence_length][
            ['GPS_MSL_Alt', 'Drift', 'Pitch', 'Roll', 'Vert_Velocity']].values

        # Fetch the validation height (target), just for printing
        validation_height = self.data_df.iloc[start_idx + sequence_length - 1]['validation_height']

        # Print the new center coordinates for debugging
        # print(f"New center coordinates in resized image: {resized_center_coords}")

        return image_sequence, flight_data, validation_height, resized_center_coords

if __name__ == "__main__":
    # get_ipython().system('pip install azure-storage-blob azure-identity --quiet')
    # get_ipython().run_line_magic('cd', '/content/drive/MyDrive/harvard.dce.nasa.cloud2cloud')
    #
    # from google.colab import drive
    #
    # drive.mount('/content/drive')


    from google.colab import userdata

    account_name = userdata.get('storage_account_name')
    account_key = userdata.get('storage_account_key')
    container_name = userdata.get('blob_container_name')

    from azure.storage.blob import BlobServiceClient

    # more info https://learn.microsoft.com/en-us/python/api/overview/azure/storage-blob-readme?view=azure-python
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    train_dates = ["20170418"] # "20170422", "20170508", "20170512"
    train_dataset = CloudDataset(train_dates, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16)

    train_dataset.data_df.head(20)
    train_dataset.data_df.tail(20)


