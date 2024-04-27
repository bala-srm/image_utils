from pathlib import Path
import shutil
import numpy as np

def split_dataset(source_dir: str, test_size: float = 0.2, val_split: float = 0.2, random_state: int = 42):
    """
    Splits a dataset into training, validation, and testing sets.

    Parameters:
    source_dir (str): The directory that contains the class directories.
    test_size (float): The proportion of the dataset to include in the test split (default 0.2).
    val_split (float): The proportion of the training dataset to include in the validation split (default 0.2).
    random_state (int): The seed used by the random number generator (default 42).
    """

    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Define the source directory and the destination directories
    src_dir = Path(source_dir)
    train_dir = src_dir / 'train'
    val_dir = src_dir / 'val'
    test_dir = src_dir / 'test'

    # Get the list of all classes
    classes = [d for d in src_dir.iterdir() if d.is_dir()]

    # For each class, split the images into training, validation, and testing
    for cls in classes:
        # Get the list of all images
        images = list(cls.iterdir())

        # Randomly shuffle the images
        np.random.shuffle(images)

        # Split the images into training, validation, and testing
        train_images = images[:int((1 - test_size - val_split) * len(images))]
        val_images = images[int((1 - test_size - val_split) * len(images)):int((1 - test_size) * len(images))]
        test_images = images[int((1 - test_size) * len(images)):]

        # For each training image, move it to the training directory
        for img in train_images:
            # Ensure the destination directory exists
            (train_dir / cls.name).mkdir(parents=True, exist_ok=True)
            # Move the image
            img.rename(train_dir / cls.name / img.name)

        # For each validation image, move it to the validation directory
        for img in val_images:
            # Ensure the destination directory exists
            (val_dir / cls.name).mkdir(parents=True, exist_ok=True)
            # Move the image
            img.rename(val_dir / cls.name / img.name)

        # For each testing image, move it to the testing directory
        for img in test_images:
            # Ensure the destination directory exists
            (test_dir / cls.name).mkdir(parents=True, exist_ok=True)
            # Move the image
            img.rename(test_dir / cls.name / img.name)