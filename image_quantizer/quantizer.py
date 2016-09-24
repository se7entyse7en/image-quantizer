from itertools import groupby
from operator import attrgetter

import numpy as np

import scipy.misc

from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

import matplotlib.pyplot as plt


def compare(*quantized_images):
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

    def __init__(self, original_raster, quantized_raster, method, n_colors,
                 **kwargs):
        self._original_raster = original_raster
        self._quantized_raster = quantized_raster
        self._method = method
        self._n_colors = n_colors
        self._kwargs = kwargs

    def render(self, show=True, new_figure=True):
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

    def __init__(self):
        self._latest_quantized_images = []

    def quantize(self, n_colors, method=None, raster=None, image_filename=None,
                 **kwargs):
        if raster is None and image_filename is None:
            raise TypeError('At least `raster` or `image_filename` must be '
                            'defined')

        if raster is None:
            raster = scipy.misc.imread(image_filename) / 255.0

        method = (method or self._method).lower()

        concrete_quantizer = {
            'random': RandomQuantizer,
            'kmeans': KMeansQuantizer,
        }[method]

        quantized_raster = concrete_quantizer.quantize(
            raster, n_colors, **kwargs)

        return QuantizedImage(
            raster, quantized_raster, method, n_colors, **kwargs)

    def quantize_multi(self, quantization_params, raster=None,
                       image_filename=None):
        return [
            self.quantize(qp['n_colors'], method=qp['method'], raster=raster,
                          image_filename=image_filename)
            for qp in quantization_params
        ]


class IConcreteQuantizer(object):

    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        raise NotImplementedError()

    @classmethod
    def _recreate_image(cls, palette, labels, width, height):
        return np.reshape(palette[labels], (width, height, palette.shape[1]))


class RandomQuantizer(IConcreteQuantizer):

    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        width, height, depth = raster.shape
        reshaped_raster = np.reshape(raster, (width * height, depth))

        palette = shuffle(reshaped_raster)[:n_colors]
        labels = pairwise_distances_argmin(reshaped_raster, palette)

        quantized_raster = cls._recreate_image(palette, labels, width, height)

        return quantized_raster


class KMeansQuantizer(IConcreteQuantizer):

    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        width, height, depth = raster.shape
        reshaped_raster = np.reshape(raster, (width * height, depth))

        model = cluster.KMeans(n_clusters=n_colors)
        labels = model.fit_predict(reshaped_raster)
        palette = model.cluster_centers_

        quantized_raster = cls._recreate_image(palette, labels, width, height)

        return quantized_raster
