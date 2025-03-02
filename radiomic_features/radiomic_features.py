# python import
from typing import Union
# package import
import SimpleITK as sitk
import pandas as pd
import numpy as np
from skimage import measure
from radiomics import featureextractor
# local import


__all__ = ['image2mask', 'extract_features']


def image2mask(img: sitk.Image, threshold: Union[int, float], ) -> sitk.Image:
    """
    Convert an input SimpleITK image to a binary mask image.

    This function performs the following steps:
    1. Binarize the image using a threshold.
    2. Generate seeds using a connected threshold algorithm.
    3. Remove small spots using morphological closing.
    4. Label connected components and retain only the largest component as the mask.

    Parameters:
        img (sitk.Image): Input SimpleITK image.
        threshold (int, float): Threshold value for binarization. If a float, it is interpreted as a percentile.

    Returns:
        sitk.Image: Binary mask image.

    Example:
        >>> img_path = '/home/user2/data/HCC-WCH/old/3-1.nii.gz'
        >>> image = sitk.ReadImage(img_path)
        >>> mask = image2mask(image, 0.05)
        >>> roi_values = sitk.GetArrayFromImage(sitk.Mask(image, mask)).flatten()
        >>> roi_values = roi_values[roi_values > 0]
        >>> roi_values = (roi_values - np.min(roi_values)) / (np.max(roi_values) - np.min(roi_values))

        >>> n = len(roi_values)
        >>> sigma = np.std(roi_values)
        >>> iqr = np.percentile(roi_values, 75) - np.percentile(roi_values, 25)
        >>> data_range = np.max(roi_values) - np.min(roi_values)

        >>> # Scott's Rule
        >>> bin_width_scott = 3.5 * sigma / (n ** (1/3))
        >>> bin_number_scott = int(np.ceil(data_range / bin_width_scott))

        >>> # Freedman-Diaconis Rule
        >>> bin_width_fd = 2 * iqr / (n ** (1/3))
        >>> bin_number_fd = int(np.ceil(data_range / bin_width_fd))

        >>> # 输出结果
        >>> print(f"Scott's Rule: bin_width = {bin_width_scott}, bin_number = {bin_number_scott}")
        >>> print(f"Freedman-Diaconis Rule: bin_width = {bin_width_fd}, bin_number = {bin_number_fd}")
    """
    # binarize the image to threshold
    vol_array = sitk.GetArrayFromImage(img)
    if isinstance(threshold, float):
        threshold = np.percentile(vol_array, 100 * (1 - threshold))

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
    morph_filter.SetKernelRadius(5)
    morph_filter.SetForegroundValue(1)
    spotless_img = morph_filter.Execute(threshold_img)

    # Extract the area (number of pixels) of each connected component
    spotless_array = sitk.GetArrayFromImage(spotless_img)
    label = measure.label(spotless_array, connectivity=2)
    props = measure.regionprops(label)
    num_pix = [prop.area for prop in props]

    if len(num_pix) == 0:
        raise ValueError("No connected components found in the mask.")

    # Retain only the largest connected component and set others to 0
    largest_label = np.argmax(num_pix) + 1  # Get the label of the largest component
    label[label != largest_label] = 0  # Set non-largest components to 0
    label[label == largest_label] = 1  # Set the largest component to 1
    label = label.astype("int16")

    if np.sum(label) == 1:
        raise ValueError("mask only contains 1 segmented voxel! Cannot extract features for a single voxel.")

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
    mask = image2mask(image, threshold)
    features = extractor.execute(image, mask)
    features['path'] = file_path  # Add image path as identifier
    logger.info(f"{file_path} extracted")
    return pd.DataFrame([features])  # Convert features to DataFrame
