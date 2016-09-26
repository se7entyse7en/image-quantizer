import os
import unittest
from unittest import mock

from image_quantizer import quantizer


class ImageQuantizerTestCase(unittest.TestCase):

    def _get_image_path(self, filename):
        return os.path.join(os.path.dirname(__file__), 'fixtures', filename)

    def test_quantize(self):
        q = quantizer.ImageQuantizer()

        q.quantize(n_colors=8, method='random',
                   image_filename=self._get_image_path('Lenna.png'))
        q.quantize(n_colors=8, method='kmeans',
                   image_filename=self._get_image_path('Lenna.png'))

    def test_quantize_multi(self):
        q = quantizer.ImageQuantizer()

        q.quantize_multi([
            {'n_colors': 8, 'method': 'random'},
            {'n_colors': 8, 'method': 'kmeans'},
        ], image_filename=self._get_image_path('Lenna.png'))

    def test_compare(self):
        q = quantizer.ImageQuantizer()

        qimages = q.quantize_multi([
            {'n_colors': 8, 'method': 'random'},
            {'n_colors': 16, 'method': 'random'},
            {'n_colors': 8, 'method': 'kmeans'},
            {'n_colors': 16, 'method': 'kmeans'},
        ], image_filename=self._get_image_path('Lenna.png'))

        with mock.patch('image_quantizer.quantizer.plt.show', lambda: None):
            quantizer.compare(*qimages)
