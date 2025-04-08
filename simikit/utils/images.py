from pathlib import Path

from PIL import Image

__all__ = [
    'verify_image_path',
    'verify_image_dir',
    'load_image',
    'resize_image'
]

IMAGE_SUFFIX = ['.jpg', '.jpeg', '.png']


def verify_image_path(filepath: str | Path) -> Path:
    """Verify that the filepath is a valid image.

    Args:
        filepath (str | Path): Path to the image.

    Returns:
        Path: Path to the image.

    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.exists() is False:
        raise FileNotFoundError(f'{filepath} does not exist.')

    if not filepath.is_file():
        raise FileNotFoundError(f'{filepath} is not a file.')

    if filepath.suffix not in IMAGE_SUFFIX:
        raise ValueError(f'{filepath} is not an image.')

    return filepath


def verify_image_dir(dir_path: str | Path) -> Path:
    """Verify that the directory exists and is a directory.

    Args:
        dir_path (str | Path): Path to the directory.

    Returns:
        Path: Path to the directory.

    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if dir_path.exists() is False:
        raise FileNotFoundError(f'{dir_path} does not exist.')

    if not dir_path.is_dir():
        raise FileNotFoundError(f'{dir_path} is not a directory.')

    for file_path in dir_path.glob('*'):
        verify_image_path(file_path)

    return dir_path


def load_image(filepath: str | Path) -> Image.Image:
    """Load an image from a file.

    Args:
        filepath (str | Path): Path to the image.

    Returns:
        Image.Image: Loaded image.

    """
    filepath = verify_image_path(filepath)

    image = Image.open(filepath)

    if image.mode != 'RGB':
        return image.convert('RGB')
    else:
        return image


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resizes an image to the specified dimensions using the LANCZOS resampling
     method.

    Args:
        image (Image.Image): The input image to be resized.
        size (Tuple[int, int]): A tuple containing the desired width and height
         for the resized image.

    Returns:
        Image.Image: The resized image with the specified dimensions.

    """
    return image.resize(size, Image.Resampling.LANCZOS)
