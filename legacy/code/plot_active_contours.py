import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# Test scipy version, since active contour is only possible
# with recent scipy version
import scipy

scipy_version = list(map(int, scipy.__version__.split('.')))
new_scipy = scipy_version[0] > 0 or \
            (scipy_version[0] == 0 and scipy_version[1] >= 14)

img = data.astronaut()
img = rgb2gray(img)

s = np.linspace(0, 2 * np.pi, 400)
x = 220 + 100 * np.cos(s)
y = 100 + 100 * np.sin(s)
init = np.array([x, y]).T

if not new_scipy:
    print('You are using an old version of scipy. '
          'Active contours is implemented for scipy versions '
          '0.14.0 and above.')

if new_scipy:
    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

"""
.. image:: PLOT2RST.current_figure

Here we initialize a straight line between two points, `(5, 136)` and
`(424, 50)`, and require that the spline has its end points there by giving
the boundary condition `bc='fixed'`. We furthermore make the algorithm search
for dark lines by giving a negative `w_line` value.
"""

img = data.text()

x = np.linspace(5, 424, 100)
y = np.linspace(136, 50, 100)
init = np.array([x, y]).T

if new_scipy:
    snake = active_contour(gaussian(img, 1), init, bc='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()

"""
.. image:: PLOT2RST.current_figure
"""
