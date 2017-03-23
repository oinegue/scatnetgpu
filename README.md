# Description

`scatnetgpu` is a module to compute the Scattering Network representation [1] of an image using the power of a CUDA capable GPU.

The output of the computation is compatible with the software [scatnet](https://github.com/scatnet/scatnet).

# Requirements

 - Python 2.7 (not tested yet with Python 3.x)
 - A CUDA capable GPU with [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) correctly installed

# Installation

Just run:
```bash
$ pip install scatnetgpu
```

Alternatively, clone the repository and run:
```bash
$ pip install .
```

# Quickstart

Load or create an image as a Numpy's ndarray. Here we load an image with OpenCV:
```python
import cv2
img = cv2.imread('image.png')
```

Create the ScatNet object:
```python
from scatnetgpu import ScatNet
sn = ScatNet(M=2, J=4, L=6)
```

Perform the actual transformation
```python
out = sn.transform(img)
```

Now `out` contains the features of the Scattering Network representation of `img`.

# References

[1] Bruna, Joan, and Stéphane Mallat. "Invariant scattering convolution networks." IEEE transactions on pattern analysis and machine intelligence 35.8 (2013): 1872-1886.