
#adjusted from the original solution by solivr by the 2D kernel into two 1D kernels

from typing import Tuple
import numpy as np

def Hessian2D(I: np.ndarray, Sigma=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function Hessian2 filters the image with 2nd derivatives of a
    Gaussian with parameter Sigma.
    :param I: image, in flotaing point precision (float64)
    :param Sigma: sigma of the gaussian kernel used
    :return: the 2nd derivatives
    """
    # Make kernel coordinates
    x = np.arange(-np.round(3*Sigma), np.round(3*Sigma) +1)
    gauss_d1 = (1 / np.sqrt(2*np.pi*Sigma**2)) * np.exp(-(x**2)/(2*Sigma**2)) * (-x/Sigma**2)
    # Build the gaussian 2nd derivatives filters
    Dx = np.convolve(I.ravel(), gauss_d1, mode='same').reshape(I.shape)
    Dy = np.convolve(I.T.ravel(), gauss_d1, mode='same').reshape(I.T.shape).T

    Dxx = np.convolve(Dx.ravel(), gauss_d1, mode='same').reshape(I.shape)
    Dxy = np.convolve(Dx.T.ravel(), gauss_d1, mode='same').reshape(I.T.shape).T
    Dyy = np.convolve(Dy.T.ravel(), gauss_d1, mode='same').reshape(I.T.shape).T
    
    return Dxx, Dxy, Dyy
