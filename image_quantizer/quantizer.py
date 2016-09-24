"""
.. module:: image_quantizer.quantizer
   :synopsis: Module for performing color quantization using different methods

"""
from itertools import groupby
from operator import attrgetter

import numpy as np

import scipy.misc

from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

import matplotlib.pyplot as plt


def compare(*quantized_images):
    """Compare multiple :class:`QuantizedImage` by showing them

    The quantized images are shown using a matrix disposition. Each row
    correspond to a method and the quantized images are sorted by increasing
    number of color used.

    :param *QuantizedImage quantized_images: the quantized images to compare.

    """
    plt.figure()

    grouped_qimages = dict((k, list(v)) for k, v in groupby(
        sorted(quantized_images, key=attrgetter('_method')),
        key=attrgetter('_method')))
    n_rows = len(grouped_qimages)
    n_cols = max(len(list(x)) for x in grouped_qimages.values())

    counter = 1
    for method, qimages in grouped_qimages.items():
        for qi in qimages:
            plt.subplot(n_rows, n_cols, counter)
            qi.render(show=False, new_figure=False)
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

        :param :class:`numpy.ndarray` original_raster: the raster of the
                                                       original image in RGB
                                                       with values normalized
                                                       in [0, 1] and shape
                                                       (width, height, 3).
        :param :class:`numpy.ndarray` quantized_raster: the raster of the
                                                        quantized image in RGB
                                                        with values normalized
                                                        in [0, 1] and shape
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
    quantizer (subclass of :class:`IConcreteQuantizer`) by calling its
    `quantize` method.

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
        :param :class:`numpy.ndarray` raster: the raster of the image in RGB
                                              with values in [0, 255] and shape
                                              (width, height, 3).
        :param str image_filename: the path of the image to quantize.
        :param **dict kwargs: extra parameters to use for the color
                              quantization.

        :raises TypeError: if the both :param:`raster` and
                           :param:`image_filename` are `None`.

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
        :param :class:`numpy.ndarray` raster: the raster of the image in RGB
                                              with values in [0, 255] and shape
                                              (width, height, 3).
        :param str image_filename: the path of the image to quantize.

        """
        return [
            self.quantize(qp['n_colors'], method=qp['method'], raster=raster,
                          image_filename=image_filename)
            for qp in quantization_params
        ]


class IConcreteQuantizer(object):
    """Interface of a concrete quantizer

    Each subclass must implement the :meth:`IConcreteQuantizer.quantize`
    method.

    """
    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        """Quantizes the given raster using the given parameters

        :param :class:`numpy.ndarray` raster: the raster of the image in RGB
                                              with values normalized in in
                                              [0, 1] and shape (width, height,
                                              3).
        :param int n_colors: the number of colors to use for the color
                             quantization.

        """
        raise NotImplementedError()

    @classmethod
    def _recreate_image(cls, palette, labels, width, height):
        return np.reshape(palette[labels], (width, height, palette.shape[1]))


class RandomQuantizer(IConcreteQuantizer):
    """Concrete quantizer that randomly selects the color palette"""

    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        width, height, depth = raster.shape
        reshaped_raster = np.reshape(raster, (width * height, depth))

        palette = shuffle(reshaped_raster)[:n_colors]
        labels = pairwise_distances_argmin(reshaped_raster, palette)

        quantized_raster = cls._recreate_image(palette, labels, width, height)

        return quantized_raster


class KMeansQuantizer(IConcreteQuantizer):
    """Concrete quantizer that selects the color palette using K-means

    The K-means algorithm is run over the pixel of the images using `k`
    clusters where `k` is the number of colors. At the end of the clustering
    the color palette will be composed by the centroids of model.

    """
    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        width, height, depth = raster.shape
        reshaped_raster = np.reshape(raster, (width * height, depth))

        model = cluster.KMeans(n_clusters=n_colors)
        labels = model.fit_predict(reshaped_raster)
        palette = model.cluster_centers_

        quantized_raster = cls._recreate_image(palette, labels, width, height)

        return quantized_raster
