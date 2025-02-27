# python import
import os
from functools import partial
from multiprocessing import Pool
# package import
import SimpleITK as sitk
import pandas as pd
import numpy as np
from skimage import measure
from radiomics import featureextractor
# local import
from utils import logger

__all__ = ['img2mask']


def img2mask(img: sitk.Image, threshold: int, ) -> sitk.Image:
    """
    Convert an input SimpleITK image to a binary mask image.

    This function performs the following steps:
    1. Binarize the image using a threshold.
    2. Generate seeds using a connected threshold algorithm.
    3. Remove small spots using morphological closing.
    4. Label connected components and retain only the largest component as the mask.

    Parameters:
        img (sitk.Image): Input SimpleITK image.
        threshold (int): Threshold value.

    Returns:
        sitk.Image: Binary mask image.
    """
    # binarize the image to threshold
    vol_array = sitk.GetArrayFromImage(img)
    vol_array = np.where(vol_array > threshold, 1, 0)
    threshold_img = sitk.GetImageFromArray(vol_array)
    threshold_img.SetSpacing(img.GetSpacing())

    # seed generation algorithm
    seed_filter = sitk.ConnectedThresholdImageFilter()
    seed_filter.SetLower(0)
    seed_filter.SetUpper(0)
    seed_filter.SetSeedList([(0, 0, 0), (img.GetSize()[0] - 1, img.GetSize()[1] - 1, 0)])
    threshold_img = seed_filter.Execute(threshold_img)
    threshold_img = sitk.ShiftScale(threshold_img, -1, -1)

    # Remove small spots using morphological closing
    morph_filter = sitk.BinaryMorphologicalClosingImageFilter()
    morph_filter.SetKernelType(sitk.sitkBall)
    morph_filter.SetKernelRadius(2)
    morph_filter.SetForegroundValue(1)
    spotless_img = morph_filter.Execute(threshold_img)

    # Extract the area (number of pixels) of each connected component
    spotless_array = sitk.GetArrayFromImage(spotless_img)
    label = measure.label(spotless_array, connectivity=2)
    props = measure.regionprops(label)
    num_pix = [prop.area for prop in props]
    # Retain only the largest connected component and set others to 0
    largest_label = np.argmax(num_pix) + 1  # Get the label of the largest component
    label[label != largest_label] = 0  # Set non-largest components to 0
    label[label == largest_label] = 1  # Set the largest component to 1
    label = label.astype("int16")

    mask_image = sitk.GetImageFromArray(label)
    mask_image.SetSpacing(img.GetSpacing())
    mask_image.SetOrigin(img.GetOrigin())  # Ensure the mask has the same origin as the image
    mask_image.SetDirection(img.GetDirection())  # Ensure the mask has the same direction as the image

    return mask_image


def extract_features(file_path: str, settings: dict, image_types: list, threshold: int, logger) -> pd.DataFrame:
    """
    Extract radiomics features for a single image and mask.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    for t in image_types:
        extractor.enableImageTypeByName(t)
    logger.info(f"{file_path} start to extract features")
    image = sitk.ReadImage(file_path)
    mask = img2mask(image, threshold)
    features = extractor.execute(image, mask)
    features['path'] = file_path  # Add image path as identifier
    return pd.DataFrame([features])  # Convert features to DataFrame


if __name__ == "__main__":
    metadata_path = "/home/user2/data/HCC-WCH/preprocessed/metadata.csv"
    metadata = pd.read_csv(metadata_path)

    settings = {
        'binWidth': 10,  # defined by intensity range of the image or normalizedScale.
        'sigma': [3, 5],
        'resampledPixelSpacing': [1, 1, 1],
        'voxelArrayShift': 0,  # whether to shift the image to > 0
        'normalize': True,
        'normalizeScale': 100,
    }

    # image_types = ['Original']
    image_types = ['LoG', 'Wavelet']
    features_save_path = f"./{image_types}_radiomics_features.csv"

    # Prepare arguments for multiprocessing
    file_paths = metadata['path'].tolist()
    partial_extract_features = partial(extract_features, settings=settings,
                                       image_types=image_types, threshold=15, logger=logger)

    # Use multiprocessing to extract features in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(partial_extract_features, file_paths)

    # Combine all DataFrames into one
    features_df = pd.concat(results, ignore_index=True)

    # Save features to CSV file
    features_df.to_csv(features_save_path, index=False)
    logger.info(f"Radiomics features saved to {features_save_path}")
