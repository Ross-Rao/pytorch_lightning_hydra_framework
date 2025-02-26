import logging
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
__all__ = ["histgram"]


def histgram(np_array, output_path, mask=None, bins=10,
             title="Gray Level Histogram", x_label="Gray Level", y_label="Count"):
    """
    Generate and save a histogram of the gray levels in a numpy array.

    Parameters:
        np_array (numpy.ndarray): Input array containing gray levels.
        output_path (str): Path to save the histogram image.
        mask (numpy.ndarray, optional): Binary mask to apply to the input array. Default is None.
        bins (int, optional): Number of bins for the histogram. Default is 10.
        title (str, optional): Title of the histogram. Default is "Gray Level Histogram".
        x_label (str, optional): Label for the x-axis. Default is "Gray Level".
        y_label (str, optional): Label for the y-axis. Default is "Count".

    Raises:
        ValueError: If the mask shape does not match the input array shape.

    Returns:
        None

    Example:
        >>> import SimpleITK as sitk
        >>> from radiomic_features.radiomic_features import img2mask
        >>> path = "/home/user2/data/HCC-WCH/old/3-1.nii.gz"
        >>> img = sitk.ReadImage(path)
        >>> img_array = sitk.GetArrayFromImage(img)
        >>> mask_img = img2mask(img, threshold=15)
        >>> mask_array = sitk.GetArrayFromImage(mask_img)
        >>> histgram(img_array, mask=mask_array, output_path='hist.png', bins=20)
    """
    plt.figure(figsize=(10, 6))

    if mask is not None:
        if mask.shape != np_array.shape:
            raise ValueError("Mask shape must match np_array shape.")
        np_array = np_array[mask > 0]

    counts, bin_edges, bars = plt.hist(np_array.flatten(), bins=bins, range=(np_array.min(), np_array.max()),
                                       color='gray', alpha=0.75)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(count)}",
                 ha="center", va="bottom", fontsize=8)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Gray level histogram saved to {output_path}")
