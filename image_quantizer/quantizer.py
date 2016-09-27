"""
.. module:: image_quantizer.quantizer
   :synopsis: Module for performing color quantization using different methods

"""
from itertools import groupby
from operator import attrgetter

import numpy as np

import scipy.misc

import matplotlib.pyplot as plt

from image_quantizer.concrete_quantizer import KMeansQuantizer
from image_quantizer.concrete_quantizer import RandomQuantizer


def _all_equal(iterator):
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


def compare(*quantized_images):
    """Compare multiple :class:`QuantizedImage` by showing them

    The quantized images are shown using a matrix disposition. Each row
    correspond to a method and the quantized images are sorted by increasing
    number of color used. The first row contains the original image.

    :param *QuantizedImage quantized_images: the quantized images to compare.

    :raises RuntimeError: if the oritinal images associated to the
                          `QuantizedImage`s are not the same.

    """
    original_rasters = [x._original_raster for x in quantized_images]

    if _all_equal(original_rasters) > 1:
        raise RuntimeError('Cannot compare multiple quantized images of '
                           'different images at the same time.')

    original_raster = original_rasters[0]

    plt.figure()

    grouped_qimages = dict((k, list(v)) for k, v in groupby(
        sorted(quantized_images, key=attrgetter('_method')),
        key=attrgetter('_method')))
    n_rows = len(grouped_qimages) + 1
    n_cols = max(len(list(x)) for x in grouped_qimages.values())

    plt.subplot(n_rows, 1, 1)
    plt.title('original')
    plt.imshow(original_raster)
    plt.axis('off')
    plt.draw()

    for i, (method, qimages) in enumerate(grouped_qimages.items(), 1):
        counter = i * n_cols + 1
        for qi in qimages:
            plt.subplot(n_rows, n_cols, counter)
            qi.render(show=False, new_figure=False)
            plt.axis('off')
            counter += 1

    plt.show()


class QuantizedImage(object):
    """Quantized image

    Together with the raster of the quantized image it also contains the raster
    of the orignal image and he parameters used to perform the color
    quantization.

    """
    def __init__(self, original_raster, quantized_raster, method, n_colors,
                 **kwargs):
        """Initializes a :class:`QuantizedImage`

        :param numpy.ndarray original_raster: the raster of the original image
                                              in RGB with values normalized
                                              in [0, 1] and shape (width,
                                              height, 3).
        :param numpy.ndarray quantized_raster: the raster of the quantized
                                               image in RGB with values
                                               normalized in [0, 1] and shape
                                               (width, height, 3).
        :param str method: the name of the method used for the color
                           quantization.
        :param int n_colors: the number of colors used to obtain the quantized
                             image.
        :param **dict kwargs: extra parameters used for the color quantization.

        """
        self._original_raster = original_raster
        self._quantized_raster = quantized_raster
        self._method = method
        self._n_colors = n_colors
        self._kwargs = kwargs

    def render(self, show=True, new_figure=True):
        """Render the quantized image

        :param bool show: if the quantized image is also to be shown and not
                          only drawn.
        :param bool new_figure: if a new figure is to be used.

        """
        if new_figure:
            plt.figure()
            plt.clf()

        plt.title('{method} ({n_colors})'.format(
            method=self._method, n_colors=self._n_colors))
        plt.imshow(self._quantized_raster)

        plt.draw()
        if show:
            plt.show()


class ImageQuantizer(object):
    """The image color quantizer

    Depending on the method to use this class uses the proper concrete
    quantizer (subclass of
    :class:`image_quantizer.concrete_quantizer.IConcreteQuantizer`) by calling
    its `quantize` method.

    """
    method_choices = {
        'random': RandomQuantizer,
        'kmeans': KMeansQuantizer,
    }

    def __init__(self, default_method=None):
        """Initializes a :class:`ImageQuantizer`

        :param str method: the name of the method to use by default for the
                           color quantization.

        """
        self._default_method = default_method

    def quantize(self, n_colors, method=None, raster=None, image_filename=None,
                 **kwargs):
        """Quantizes the given image or raster using the given parameters

        :param int n_colors: the number of colors to use for the color
                             quantization.
        :param str method: the name of the method to use for the color
                           quantization.
        :param numpy.ndarray raster: the raster of the image in RGB with values
                                     in [0, 255] and shape (width, height, 3).
        :param str image_filename: the path of the image to quantize.
        :param **dict kwargs: extra parameters to use for the color
                              quantization.

        :raises TypeError: if the both `raster` and `image_filename` are
                           `None`.

        """
        if raster is None and image_filename is None:
            raise TypeError('At least `raster` or `image_filename` must be '
                            'defined')

        if raster is None:
            raster = scipy.misc.imread(image_filename) / 255.0

        method = (method or self._default_method).lower()

        concrete_quantizer = self.method_choices[method]

        quantized_raster = concrete_quantizer.quantize(
            raster, n_colors, **kwargs)

        return QuantizedImage(
            raster, quantized_raster, method, n_colors, **kwargs)

    def quantize_multi(self, quantization_params, raster=None,
                       image_filename=None):
        """Quantizes the given image or raster using mutilpe configurations

        :param list quantization_param: the list of configurations to use for
                                        the color quantization where each
                                        element is a dictionary with keys
                                        `n_colors` and `method` (see
                                        :meth:`ImageQuantizer.quantize`).
        :param numpy.ndarray raster: the raster of the image in RGB with values
                                     in [0, 255] and shape (width, height, 3).
        :param str image_filename: the path of the image to quantize.

        """
        return [
            self.quantize(qp['n_colors'], method=qp['method'], raster=raster,
                          image_filename=image_filename)
            for qp in quantization_params
        ]
