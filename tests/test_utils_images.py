from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from simikit.utils.images import IMAGE_SUFFIX, load_image, resize_image, verify_image_dir, verify_image_path


class TestVerifyImagePath:
    def test_valid_image_path(self):
        """Test a valid image path."""
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'is_file', return_value=True):
                for suffix in IMAGE_SUFFIX:
                    file_path = f'/path/to/image{suffix}'

                    # in str
                    result = verify_image_path(file_path)

                    assert result == Path(file_path)

                    # in Path
                    result = verify_image_path(Path(file_path))

                    assert result == Path(file_path)

    def test_file_not_found(self):
        """Test the case where the file does not exist."""
        with patch.object(Path, 'exists', return_value=False):
            file_path = Path('/path/to/nonexistent_image.jpg')
            with pytest.raises(FileNotFoundError):
                verify_image_path(file_path)

    def test_not_a_file(self):
        """Test the case where the path does not point to a file."""
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'is_file', return_value=False):
                file_path = Path('/path/to/directory')
                with pytest.raises(FileNotFoundError):
                    verify_image_path(file_path)

    def test_not_an_image(self):
        """Test the case where the path does not point to an image file."""
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'is_file', return_value=True):
                file_path = Path('/path/to/document.txt')
                with pytest.raises(ValueError):
                    verify_image_path(file_path)


class TestVerifyImageDir:
    def test_valid_image_dir(self):
        """Test a valid image directory."""
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'is_dir', return_value=True):
                mock_path_list = [Path(f'/path/to/image{ext}') for ext in IMAGE_SUFFIX]
                with patch.object(Path, 'glob', return_value=mock_path_list):
                    with patch('simikit.utils.images.verify_image_path') as mock_verify_image_path:
                        dir_path = '/path/to/image_dir'

                        # in str
                        result = verify_image_dir(dir_path)

                        assert result == Path(dir_path)
                        assert mock_verify_image_path.call_count == len(IMAGE_SUFFIX)

                        # in Path
                        mock_verify_image_path.reset_mock()
                        result = verify_image_dir(Path(dir_path))

                        assert result == Path(dir_path)
                        assert mock_verify_image_path.call_count == len(IMAGE_SUFFIX)

    def test_dir_not_found(self):
        """Test the case when the directory does not exist."""
        with patch.object(Path, 'exists', return_value=False):
            dir_path = Path('/nonexistent/dir/path')
            with pytest.raises(FileNotFoundError):
                verify_image_dir(dir_path)

    def test_not_a_dir(self):
        """Test the case when the path is not a directory."""
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'is_dir', return_value=False):
                dir_path = Path('/not/a/dir/path')
                with pytest.raises(FileNotFoundError):
                    verify_image_dir(dir_path)

    def test_verify_image_path_raises_error(self):
        """Test the case when the verify_image_path function raises an error for a file in the directory."""
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'is_dir', return_value=True):
                with patch.object(Path, 'glob', return_value=[Path('file1.jpg')]):
                    with patch('simikit.utils.images.verify_image_path', side_effect=ValueError):
                        dir_path = Path('/valid/dir/path')
                        with pytest.raises(ValueError):
                            verify_image_dir(dir_path)


class TestLoadImage:
    def test_load_image(self):
        """Test the load_image function with a mock RGB image"""
        mock_file_path = Path('/path/to/image.jpg')
        mock_image = Image.new('RGB', (100, 100))
        with patch('simikit.utils.images.verify_image_path', return_value=mock_file_path):
            with patch('PIL.Image.open', return_value=mock_image):
                result = load_image(mock_file_path)

                assert isinstance(result, Image.Image)
                assert result.mode == 'RGB'

    def test_load_image_convert(self):
        """Test the image mode conversion functionality in the load_image function."""
        mock_file_path = Path('/path/to/image.jpg')
        mock_image = Image.new('L', (100, 100))
        with patch('simikit.utils.images.verify_image_path', return_value=mock_file_path):
            with patch('PIL.Image.open', return_value=mock_image):
                result = load_image(mock_file_path)

                assert isinstance(result, Image.Image)
                assert result.mode == 'RGB'


def test_resize_image():
    """Test the resize_image function."""
    mock_image = Image.new('RGB', (100, 100))
    target_size = (200, 200)
    with patch.object(mock_image, 'resize') as mock_resize:
        resize_image(mock_image, target_size)
        mock_resize.assert_called_once_with(target_size, Image.Resampling.LANCZOS)
