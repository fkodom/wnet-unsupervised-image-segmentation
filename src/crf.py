r"""
crf.py
---------
Implements a Conditional Random Field (CRF) for image segmentation, using the `pydensecrf` library.
"""

import numpy as np
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


def crf_fit_predict(softmax: np.ndarray, image: np.ndarray, niter: int = 150):
    r"""Fits a Conditional Random Field (CRF) for image segmentation, given a mask of class probabilities (softmax)
    from the WNet CNN and the raw image (image).

    :param softmax: Softmax outputs from a CNN segmentation model.  Shape: (nchan, nrow, ncol)
    :param image: Raw image, containing any number of channels.  Shape: (nchan, nrow, ncol)
    :param niter: Number of iterations during CRF optimization
    :return: Segmented class probabilities -- take argmax to get discrete class labels.
    """
    unary = unary_from_softmax(softmax).reshape(softmax.shape[0], -1)
    bilateral = create_pairwise_bilateral(sdims=(25, 25), schan=(0.05, 0.05), img=image, chdim=0)

    crf = dcrf.DenseCRF2D(image.shape[2], image.shape[1], softmax.shape[0])
    crf.setUnaryEnergy(unary)
    crf.addPairwiseEnergy(bilateral, compat=100)
    pred = crf.inference(niter)

    return np.array(pred).reshape((-1, image.shape[1], image.shape[2]))


def crf_batch_fit_predict(probabilities: np.ndarray, images: np.ndarray, niter: int = 150):
    r"""Fits a Conditional Random Field (CRF) for image segmentation, given a mask of class probabilities (softmax)
    from the WNet CNN and the raw image (image).

    :param probabilities: Softmax outputs from a CNN segmentation model.  Shape: (batch, nchan, nrow, ncol)
    :param images: Raw image, containing any number of channels.  Shape: (batch, nchan, nrow, ncol)
    :param niter: Number of iterations during CRF optimization
    :return: Segmented class probabilities -- take argmax to get discrete class labels.
    """
    return np.stack([crf_fit_predict(p, x, niter) for p, x in zip(probabilities, images)], 0)
