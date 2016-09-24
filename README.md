# Image Quantizer
## Color Image Quantization

Perform color quantization by selecting the palette randomly or using K-means.

### Environment

Setup the environment using conda:

    conda env create -f environment.yml

### Sample Usage

Here's a sample usage that displays the output of the color quantization
performed on the same image using random selection and K-means:

    from image_quantizer import quantizer

    q = quantizer.ImageQuantizer()

    out1 = q.quantize(n_colors=8, method='random', image_filename='myimage.png')
    out2 = q.quantize(n_colors=8, method='kmeans', image_filename='myimage.png')

    quantizer.compare(out1, out2)
