import cv2
import numpy as np
from scipy.spatial import ConvexHull
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from copy import deepcopy
from cv2 import resize
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon
import os
import sys
from typing import Tuple


# LANDMARKS INDEXES BOUNDARING THE REGIONS
#parts where no wrinkles are localized
left_eyebrow = [107, 66, 105, 63 , 70 , 156, 124, 113, 225, 224, 223, 222, 221, 55 ]
right_eyebrow = [368, 300, 293, 334, 296, 336, 285, 441, 442, 443, 444, 445, 353, 383]
left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
nose_tip = [2, 326, 327, 423, 358, 429, 3, 126, 203, 98, 97]
mouth = [17, 314, 405, 273, 287, 391, 326, 97, 165, 57, 43, 181, 84]
body_parts = (left_eyebrow, right_eyebrow, left_eye, right_eye, mouth)

#face parts
upper_face = [137, 93, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 323, 366]
lower_face = [137, 123, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 366, 352]
left_half = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 10, 67, 109, 10]
right_half = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152,
              377, 400, 378, 379, 365, 397, 288, 435, 361, 323, 454, 389, 251, 284, 332, 297, 338, 10]
naso_left = [174, 100, 205, 207, 214, 210, 204, 92, 98, 134]
naso_right = [399, 330, 425, 427, 434, 430, 431, 322, 305, 363]
forehead = [54, 68, 105, 66, 107, 9, 336, 296, 334, 293, 298, 284, 332, 297, 338, 10, 109, 67, 103, 54]

#!wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
if os.path.exists('dir\\face_landmarker_v2_with_blendshapes.task'):
    detector_path = 'dir\\face_landmarker_v2_with_blendshapes.task'

def obtain_google_landmarks(image, detector_path: str = detector_path):
    """
    This function returns detection results from MediaPipe face landmarks deteector
    :param image: image, uint8 np.ndarray or path to the image
    :param detector_path: path to MediaPipe detector
    :return: Mediapipe landmarks
    """
    base_options = python.BaseOptions(model_asset_path=detector_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    if type(image) == str:
        image = mp.Image.create_from_file(image)
    else:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(image)
    return detection_result


def google_lmrks2coords(landmarks, w: int = 1920, h: int = 1080) -> np.ndarray:
    """
    This function returns 478x3 numpy array with coordinates resized to image height and width
    :param landmarks: Mediapipe landmarks
    :param w: image width
    :param h: image height
    """
    coords = np.zeros((478, 3)) #478 is the number of landmarks x 3 is the (X, Y, Z)
    for i, landmark in enumerate(landmarks.face_landmarks[0]):
        coords[i, 0] = landmark.x * w
        coords[i, 1] = landmark.y * h
        coords[i, 2] = landmark.z * w
    return coords

def crop_face(img: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function crops the image to contain the face.
    :param img: original image
    :param coords: 478 coordinates of face landmarks
    :return: cropped face and adjusted coordinates
    """
    h, w, _ = img.shape
    min_x = int(np.floor(np.min(coords[:, 0]))) - 20
    max_x = int(np.ceil(np.max(coords[:, 0])) +1) + 20
    min_y = np.floor(np.min(coords[:, 1]))
    max_y = np.ceil(np.max(coords[:, 1]))
    h_face = max_y - min_y

    min_y_face_adj = int(np.floor(max(min_y - h_face*4/25, 0)))
    max_y_face_adj = int(np.ceil(min(max_y + h_face/20 + 1, h)))    

    img_cropped = img[min_y_face_adj:max_y_face_adj, min_x:max_x, :]
    coords_cropped = deepcopy(coords)
    coords_cropped[:, 0] -= min_x
    coords_cropped[:, 1] -= min_y_face_adj
    return img_cropped, coords_cropped, [min_y_face_adj, max_y_face_adj, min_x, max_x]

def show_img_with_2D_coords(img: np.ndarray, coords: np.ndarray = None, verbosity: bool = False, title: str = "") -> None:
    """
    This function shows the image with scattered coordinates
    """
    plt.imshow(img)
    if coords is not None:
        plt.scatter(coords[:, 0], coords[:, 1], s=1)
    for (i,coord) in enumerate(coords):
        plt.text(coord[0], coord[1], str(i))
    if verbosity:
        plt.show()
    plt.title(title)
    return

def _segment_face(coords, mask):
    hull = ConvexHull(coords[:, :2])
    points = np.array(coords[hull.vertices, :2], dtype=np.int32)
    print(points.shape)
    cv2.fillConvexPoly(mask, points, 255)

def _segment_non_wrinkle_parts(coords, mask):
    for part in body_parts:
        part = np.array(part)
        hull = ConvexHull(coords[part, :2])
        points = np.array(coords[part, :2][hull.vertices, :2], dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)

def resize_coords(img_source_shape, img_target_shape, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h1, w1 = img_source_shape
    h2, w2 = img_target_shape
    coords[:, 0] *= w2/w1
    coords[:, 1] *= h2/h1
    return coords

def get_naso_limits(coords) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Returns the limits for windows where nasolabial wrinkles are localized.
    :param coords: 478 coordinates of face landmarks
    :return: coordinates of pixels which can be used for cropping nasolabial regions
    """
    nr_crds = np.round(coords[naso_right, :2]).astype(np.int32)
    nl_crds = np.round(coords[naso_left, :2]).astype(np.int32)

    nr_lims = [np.min(nr_crds[:, 0]) - 5, np.max(nr_crds[:, 0]) + 5, np.min(nr_crds[:, 1]) - 5, np.max(nr_crds[:, 1]) + 5]
    nl_lims = [np.min(nl_crds[:, 0]) - 5, np.max(nl_crds[:, 0]) + 5, np.min(nl_crds[:, 1]) - 5, np.max(nl_crds[:, 1]) + 5]
    return nr_lims, nl_lims

def segment_one_part(coords, mask, part_name):
    """
    This function segments chosen face part, inplace operation.
    :param coords: 478 coordinates of face landmarks
    :param mask: numpy bool array, mask of the shape corresponding to the coordinates
    :param part_name: name of the face part to be segmented = ["upper","lower","right","left","naso_right","naso_left","forehead"]
    """
    if part_name == "upper": part = upper_face
    elif part_name == "lower": part = lower_face
    elif part_name == "left": part = left_half
    elif part_name == "right": part = right_half
    elif part_name == "naso_right": part = naso_right
    elif part_name == "naso_left": part = naso_left
    elif part_name == "forehead": part = forehead
    else: raise ValueError("Not implemented face part")
    part = np.array(part)
    hull = ConvexHull(coords[part, :2])
    hull_pts = coords[part[hull.vertices], :2]
    # minihull because the ordering of the points matters
    rr, cc = polygon(hull_pts[:, 1], hull_pts[:, 0], shape=mask.shape)
    mask[rr, cc] = True
    return #this operation is inplace

def segment(img) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Function gets input image and returns its landmarks and masks.
    :param img_path: string with absolute or relative path to your image
    :return img: (h, w, 3) 255 uint8 numpy array
    :return mask: (h, w) bool numpy array mask which contains only parts with wrinkles
    :return coords: (478, 3) numpy array with coordinates of landmarks
    :return limits: (478, 3) return incides used for cropping
    :return mask: (478, 3) convex polygon containing the face coordinates
    """
    if type(img) == str:
        image = np.array(Image.open(img))
    else:
        image = img
    coords = google_lmrks2coords(obtain_google_landmarks(img), w=image.shape[1], h=image.shape[0])
    image, coords, limits = crop_face(image, coords)
    mask_whole = np.zeros(image.shape[:2], dtype=np.uint8)
    _segment_face(coords, mask_whole) # mask is segmented in_place
    mask_partial = np.zeros(image.shape[:2], dtype=np.uint8)
    _segment_non_wrinkle_parts(coords, mask_partial)
    mask_whole = mask_whole.astype(bool)
    mask_partial = mask_partial.astype(bool)
    mask = (~mask_partial * mask_whole)
    return image, mask, coords, limits, mask_whole