"""
.. module:: image_quantizer.concrete_quantizer
   :synopsis: Module containing the concrete classes the perform the color
              quantization.

"""
import numpy as np

from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

from skimage import color


class IConcreteQuantizer(object):
    """Interface of a concrete quantizer

    Each subclass must implement the :meth:`IConcreteQuantizer.quantize`
    method.

    """
    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        """Quantizes the given raster using the given parameters

        :param numpy.ndarray raster: the raster of the image in RGB with values
                                     in [0, 255] and shape (width, height, 3).
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


class RGBtoLABmixin(object):

    @classmethod
    def quantize(cls, raster, n_colors, **kwargs):
        lab_raster = color.rgb2lab(raster)
        lab_quantized_raster = super(RGBtoLABmixin, cls).quantize(
            lab_raster, n_colors, **kwargs)
        quantized_raster = (color.lab2rgb(lab_quantized_raster) * 255).astype(
            'uint8')

        return quantized_raster


class RandomQuantizerLAB(RGBtoLABmixin, RandomQuantizer):
    pass


class KMeansQuantizerLAB(RGBtoLABmixin, KMeansQuantizer):
    pass
