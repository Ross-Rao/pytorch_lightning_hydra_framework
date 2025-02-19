import os
import pandas as pd
import re
import logging


logger = logging.getLogger(__name__)


def dir_to_df(root: str, ext: str, name_template: str = None) -> pd.DataFrame:
    """
    Converts files in a directory to a pandas DataFrame.

    Args:
        root (str): The root directory containing the files.
        ext (str): The file extension to filter files.
        name_template (str, optional): A template for extracting information from file names.

    Returns:
        pd.DataFrame: A DataFrame containing file paths and extracted information.

    Example:
        >>> # Assuming the following directory structure:
        >>> # /home/user2/data/HCC-WCH/old
        >>> # ├── 1-1.nii.gz
        >>> # ├── 1-2.nii.gz
        >>> # ├── ...

        >>> data_root = os.getenv("DATASET_LOCATION", '/home/user2/data')
        >>> wch = os.path.join(data_root, 'HCC-WCH', 'old')

        >>> df = dir_to_df(root=wch, ext=".nii.gz", name_template='{patient_number}-{label}')
    """
    assert os.path.exists(root) and os.path.isdir(root), f"Path {root} does not exist or is not a directory."

    file_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(ext)]
    file_names = [os.path.basename(f)[: -len(ext)] for f in file_paths]

    data = {'path': file_paths, 'name': file_names}

    if name_template:
        template_pattern = re.sub(r'{(\w+)}', r'(?P<\1>.+)', name_template)
        regex = re.compile(template_pattern)

        for key in regex.groupindex.keys():
            data[key] = []

        for name in file_names:
            match = regex.match(name)
            if match:
                for key, value in match.groupdict().items():
                    data[key].append(value)
            else:
                for key in regex.groupindex.keys():
                    data[key].append("")
                logger.warning(f"Name {name} does not match the template {name_template}.")

    file_df = pd.DataFrame(data)
    return file_df
