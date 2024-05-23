import numpy as np

def histogram_equalization_gray(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Performs histogram normalization of gray-scale 8-bit image, only pixels with mask 1 are taken in account
    :param img: hxw uint8 ndarray
    :param mask: hxw bool ndarray
    :return img_out: hxw ndarray with equalized histogram
    """
    img_f = img.flatten()
    if mask is None:
        mask_f = np.ones_like(img_f, dtype=bool)
    else:
        mask_f = mask.flatten()

    histogram, _ = np.histogram(img_f[mask_f], bins=256, range=(0, 255))
    histogram = histogram / img.shape[0]
    cdf = histogram.cumsum()
    cdf_norm = 255*(((cdf - cdf.min())) / (cdf.max() - cdf.min()))
    cdf_norm = cdf_norm.astype(np.uint8)

    img_out = cdf_norm[img_f]
    img_out = np.reshape(img_out, img.shape)
    return img_out

def histogram_equalization_RGB(img: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Performs histogram normalization of RGB 8-bit image channel-wise, only pixels with mask 1 are taken in account
    :param img: hxw uint8 ndarray
    :param mask: hxw bool ndarray
    :return img_out: hxw ndarray with equalized histogram
    """
    img_out = np.zeros_like(img, dtype='uint8')
    for i in range(3):
        img_out[:, :, i] = histogram_equalization_gray(img[:, :, i], mask)
    return img_out
    
