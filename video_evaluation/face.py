import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata, RectBivariateSpline
import queue
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.draw import polygon
from typing import Tuple, Tuple
#from frangiFilter2D import FrangiFilter2D #frangi filter can be downloaded at https://github.com/solivr/frangi_filter/blob/master/frangiFilter2D.py

from segmentation import *

face_connections = np.load("adjacency_matrix.npy") # load saved adjacency matrix


def find_affine_2D(coords_source: np.ndarray, coords_target: np.ndarray) -> np.ndarray:
    """
    Finds T estimate for 2D affine transformation by least squares methosds.
    """
    A = np.column_stack([coords_source[:, 0:2], np.ones_like(coords_source[:, 1])])
    B = coords_target[:, 0:2]
    T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return T.T
 
def transf_source2target_2D(img_source: np.ndarray, coords_source: np.ndarray, coords_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms source image into targeted image through 2D Affine Transformation.
    """
    h, w = img_source.shape[:2]
    T = find_affine_2D(coords_source, coords_target)
    coords_t = (T @ np.vstack([coords_source[:, 0:2].T, np.ones_like(coords_source[:, 0])])).T
    img_t = cv2.warpAffine(img_source, T, (w, h))
    return img_t, coords_t

def transf_source2target_piecwise(img_source: np.ndarray, coords_source: np.ndarray, coords_target:np.ndarray) -> Tuple[np.ndarray]:
    """
    Transforms source image into targeted image through Piecewise Affine Transformation.
    """
    assert coords_source.shape[:2] == coords_target.shape[:2]

    h, w = img_source.shape[:2]

    src = coords_source[:, :2]
    dst = coords_target[:, :2]
    tform = PiecewiseAffineTransform()
    tform.estimate(dst, src)
    img_t = warp(img_source, tform, output_shape=(h, w))
    return img_t

def find_inverse_affine_3D(coords_source: np.ndarray, coords_target: np.ndarray) -> np.ndarray:
    """
    Finds T^-1 estimate for 3D affine transformation by least squares methosds.
    """
    A = np.column_stack([coords_target, np.ones_like(coords_target[:, 1])])
    B = coords_source
    T, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return T.T

def transform_coords_3D(coords_source: np.ndarray, coords_target: np.ndarray) -> np.ndarray:
    """
    Transforms source coordinates into targeted coordinates through 3D transformation T.
    """
    T_inv = find_inverse_affine_3D(coords_source=coords_source, coords_target=coords_target)
    coords_t = (T_inv @ np.vstack([coords_target.T, np.ones_like(coords_target[:, 0])])).T
    return coords_t

def create_3D_grid(coords: np.ndarray, h: int = 1080, w: int = 1920, return_hull = False) -> np.ndarray:
    """
    Transforms source image into targeted image through Piecewise Affine Transformation.
    """
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]
    # querry points
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xq, yq = np.meshgrid(x, y)

    # Interpolate the irregularly spaced data onto the regular grid
    Zq = griddata((X, Y), Z, (xq, yq), method='linear')
    if return_hull == True:
        hull = 1 - np.isnan(Zq)
        hull = np.dstack((hull, hull, hull))
    Zq_nearest = griddata((X, Y), Z, (xq, yq), method='nearest')
    # set value of the points outside the convex hull to the nearest pixel of the convex hull
    Zq[np.isnan(Zq)] = Zq_nearest[np.isnan(Zq)]
    if return_hull == False:
        return Zq, None
    else:
        return Zq, hull
        
def transf_source2target_3D(img_source: np.ndarray, coords_source: np.ndarray, coords_target: np.ndarray, h: int = 1080, w: int = 1920):
    """
    Transforms source image into targeted coordinates through 3D transformation T.
    """
    h_s, w_s, c_s = img_source.shape
    X, Y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    Z_target, hull = create_3D_grid(coords_target, h, w, return_hull=True)
    T_inv = find_inverse_affine_3D(coords_source, coords_target)    
    transformed_grid = (T_inv @ np.vstack((X.flatten(), Y.flatten(), Z_target.flatten(), np.ones(h*w)))).T
    x_transformed = transformed_grid[:, 0].reshape(h_s, w_s)
    y_transformed = transformed_grid[:, 1].reshape(h_s, w_s)
    interpolators = [RectBivariateSpline(np.arange(h_s), np.arange(w_s), img_source[:, :, i], kx=1, ky=1) for i in range(c_s)]
    channel_values = [interp(y_transformed, x_transformed, grid=False) for interp in interpolators]
    image_t = np.stack(channel_values, axis=-1).astype(np.uint8)
    # transform coords
    T_inv = np.concatenate((T_inv, np.array([[0, 0, 0, 1]])), axis=0)
    T = np.linalg.inv(T_inv)
    T = T[:4, :]
    coords_t = (T @ np.vstack((coords_source.T, np.ones_like(coords_source[:, 0])))).T
    return image_t, coords_t, hull

def make_transformation_matrix(thetas: np.ndarray, scale: float = 1):
    """
    Compute 3D transformation with defined angles.
    :param thetas: (3,) numpy array of angles of rotations
    :param scale: scale of the 3D transformation
    """
    Smat = scale * np.eye(3)
    theta_x, theta_y, theta_z = thetas

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    out = Rz @ Rx @ Ry  @ Smat
    out = np.column_stack((out, np.zeros(3)))
    return out

class Vertex(object):
    def __init__(self, idx: int, x: float, y: float, z: float) -> None:
        """
        Initialization of Vertex class, vertex idx should correspond to ordering in FaceLandmarks.
        x, y, z should be rescaled so that they correspond to image width and height.
        """
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z
        self.neighs = []
        self.on_queue = False
        self.open = False
        self.on_surface = False

    def __lt__(self, other):
        # for sorting purposes
        return self.z < other.z
    
    def reset(self):
        self.on_queue = False
        self.open = False
        self.on_surface = False

def push_neighbour(v1: Vertex, v2: Vertex):
    v1.neighs.append(v2)

class FaceMesh:
    def __init__(self, coords: np.ndarray, img_w: int, img_h: int, verbosity = False) -> None:
        """
        Inicialization of FaceMesh, coords should be 478x3 array of coordinates locations in in range of image shapes.
        """
        assert (478, 3) == coords.shape
        self.coords = coords
        self.img_h = img_h
        self.img_w = img_w
        self.vertices: Tuple[Vertex] = [] #prepared space for all vertices
        self.mask = np.zeros((self.img_h, self.img_w)).astype(np.bool_)
        self.o = [] #order of assinged vertices
        self.verbosity = verbosity #show progress of vertices bfs
        self.surface_vertexes = np.zeros(len(self.vertices), dtype=np.bool_) #bool of indicating if vertex is on the surface, false at default

        # insert all vertices
        for i in range(coords.shape[0]):
            self.vertices.append(Vertex(i, coords[i, 0], coords[i, 1], coords[i, 2]))

        # make connection between connected vertices
        for i in range(coords.shape[0]):
            for j in range(coords.shape[0]):
                if face_connections[i, j]:
                    push_neighbour(self.vertices[i], self.vertices[j])

        #sort all neighbours of all vertices by their z-coordinate in ascending order

        self.sort_neighs()
        self.bfs()
        

    def sort_neighs(self) -> None:
        """Sorts neighbours of all vertices by their z-cooridnate in ascending order"""
        for vertex in self.vertices:
            vertex.neighs = sorted(vertex.neighs)

    def reset(self) -> None:
        """Set all boolean atributes of all vertices to False"""
        for vertex in self.vertices():
            vertex.reset()
                        
    def push_to_grid(self, vertex: Vertex) -> bool:
        """Pushes vertex to the image grid and returns if this vertex is on surface or not"""
        verbosity = self.verbosity      
        out = False
        if self.mask[int(round(np.clip(vertex.y, 0, self.img_h-1))), int(round(np.clip(vertex.x, 0, self.img_w-1)))] == False:
            # if the vertex being checked does not lie inside the mask
            out = True
            minihull_pts = np.array((vertex.y, vertex.x))
            for n in vertex.neighs:
                n: Vertex
                if n.on_surface:
                    minihull_pts = np.vstack((minihull_pts, (n.y, n.x)))
            if len(minihull_pts) > 2:
                # create a polygon of the vertex and its neighbours already on the surface and set mask pixels inside of it on true
                minihull = ConvexHull(minihull_pts)
                minihull_pts = minihull_pts[minihull.vertices, :]
                # minihull because the ordering of the points matters
                rr, cc = polygon(minihull_pts[:, 0], minihull_pts[:, 1], shape=self.mask.shape)
                self.mask[rr, cc] = True                

        if verbosity and len(self.o) % 50 == 0:
            #plotting
            plt.imshow(self.mask, cmap="gray")
            plt.scatter(self.coords[self.o, 0], self.coords[self.o, 1])
            plt.scatter(vertex.x, vertex.y, s=5, c='r')
            plt.legend(("Mask", "Surface points", "Candidate surface point"))
            plt.title("Success" if out else "Not success")
            plt.plot()
            plt.show()

        return out
    
    def bfs(self):
        "Method for adding vertices to surface"
        Q = queue.Queue()
        first_idx = min(range(len(self.vertices)), key=lambda i: self.vertices[i].z)
        Q.put(self.vertices[first_idx])
        
        self.vertices[first_idx].on_queue = True

        while Q.qsize() != 0:
            current_vertex: Vertex = Q.get()
            current_vertex.on_queue = False
            current_vertex.open = True
            self.o.append(current_vertex.idx)
            if self.push_to_grid(current_vertex): # this vertex mask
                current_vertex.on_surface = True
                self.surface_vertexes[current_vertex.idx] = True
                for neigh in current_vertex.neighs:
                    if not neigh.on_queue and not neigh.open: # push not explored vertices on the queue
                        Q.put(neigh)
                        neigh.on_queue = True


class Face:
    def __init__(self, img, wrk = None, coords = None, verbosity = False, is_cropped = True, copmute_mesh = False) -> None:
        """
        Inicialization of face class.
        :param img: image of the face
        :param wrk: mask of wrinkle with same shape as img
        :param coords: 478 MediaPipe face landmarks
        :param verbosity: set true to see progress of surface bfs
        :param is_cropped: is the face cropped
        :compute_mesh: set to true if face mesh should be computed
        """
        assert img.shape[:2] == wrk.shape[:2]
        self.verbosity = verbosity
        self.img = img
        self.shape = self.img.shape[:2]
        self.mesh = None
        self.wrk = wrk

        if coords is not None:
            self.coords = coords
        else:
            self.coords = google_lmrks2coords(obtain_google_landmarks(img), w=self.shape[1], h=self.shape[0])
        
        if is_cropped == False:
            self.img, self.coords, _ = crop_face(self.img, self.coords)


        
        if copmute_mesh:
            self.mesh = FaceMesh(coords=self.coords, img_w=self.img.shape[1], img_h=self.img.shape[0], verbosity=self.verbosity)

    def copmute_mesh(self):
        """
        Computes mesh containing information if landmark is hidden or not.
        """
        if self.mesh is None:
            self.mesh = FaceMesh(coords=self.coords, img_w=self.img.shape[1], img_h=self.img.shape[0], verbosity=self.verbosity)

    def show_img_with_wrinkles(self) -> None:
        """
        Displays image with wrinkle mask over it in red
        """
        img = np.copy(self.img)
        mask = self.wrk
        img[mask, :] = [255, 0, 0]
        plt.imshow(img)
        plt.show()
    
    def show_img_with_landmarks(self) -> None:
        """
        Display image with coordinates over it. Coordinates on the front surface are in blue, otherwise in red.
        """
        plt.imshow(self.img)
        plt.scatter(self.coords[self.mesh.surface_vertexes, 0], self.coords[self.mesh.surface_vertexes, 1], s=2, c='b')
        plt.scatter(self.coords[~self.mesh.surface_vertexes, 0], self.coords[~self.mesh.surface_vertexes, 1], s=2, c='r')
        plt.show()

    def show_grid(self) -> None:
        """
        Display image with grid landmarks grid. Connections between surface vertices in green, between non-surfce vertices in red, blue otherwise.
        """
        fig = plt.figure()
        plt.imshow(self.img)
        for vertex in self.mesh.vertices:
            for neigh in vertex.neighs:
                if self.mesh.surface_vertexes[vertex.idx] and self.mesh.surface_vertexes[neigh.idx]:
                    color = 'green'
                elif not self.mesh.surface_vertexes[vertex.idx] and not self.mesh.surface_vertexes[neigh.idx]:
                    color = 'red'
                else:
                    color = 'blue'
                
                plt.plot([vertex.x, neigh.x], [vertex.y, neigh.y], color=color)
        plt.show()

    def show_landmarks_development(self) -> None:
        """Shows 3D scatter plot of face coordinates."""
        for idx in range(0, len(self.mesh.o), 20):
            idxs = self.mesh.o[:idx]
            plt.imshow(self.img)
            plt.scatter(self.coords[idxs, 0], self.coords[idxs, 1])
            plt.show()

    def show_3D_coords(self, fig) -> None:
        coords_3D = self.coords
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords_3D[:, 0], coords_3D[:, 1], coords_3D[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_box_aspect([1, 1, 1])
        return

    def rotate_face(self, theta_x:float, theta_y:float, theta_z:float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes piecewise affine transformation of estimated 3D rotation.
        """
        T = make_transformation_matrix([theta_x, theta_y, theta_z])
        coords_t = transform_coords_3D(T, self.coords)
        mesh_t = FaceMesh(coords_t, self.mesh.img_w, self.mesh.img_h)

        valid_coords_source = self.coords[self.mesh.surface_vertexes & mesh_t.surface_vertexes, :]
        valid_coords_target = coords_t[self.mesh.surface_vertexes & mesh_t.surface_vertexes, :]

        img_t = transf_source2target_piecwise(self.img, valid_coords_source, valid_coords_target)
        wrk_t = transf_source2target_piecwise(self.wrk.astype(np.float32), valid_coords_source, valid_coords_target)
        wrk_t = (wrk_t/255).astype(np.bool_)

        return img_t, wrk_t

    def save(self):
        if self.save_file_img:
            plt.imsave(self.save_file_img, self.img, )
        if self.save_file_wrk:
            np.save(self.save_file_wrk, self.wrk)
        if self.save_file_crd:
            np.save(self.save_file_crd, self.coords)

    # parts segmentation
    def upper(self):
        self.mask_upper = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_upper, "upper")
    def lower(self):
        self.mask_lower = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_lower, "lower")
    def left(self):
        self.mask_left = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_left, "left")
    def right(self):
        self.mask_right = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_right, "right")
    def naso_right(self):
        self.mask_naso_right = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_naso_right, "naso_right")
    def naso_left(self):
        self.mask_naso_left = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_naso_left, "naso_left")
    def forehead(self):
        self.mask_forehead = np.zeros(self.shape)
        segment_one_part(self.coords, self.mask_forehead, "forehead")

    # parameters for single image
    def overall_wrinkles(self) -> float:
        return float(self.wrk.sum())
    def upper_to_lower(self) -> float:
        upper_sum = np.sum(self.wrk*self.mask_upper)
        lower_sum = np.sum(self.wrk*self.mask_lower)
        return float((upper_sum - lower_sum)/(lower_sum + upper_sum+1))
    def left_to_right(self) -> float:
        left_sum = np.sum(self.wrk*self.mask_left)
        right_sum = np.sum(self.wrk*self.mask_right)
        return float((left_sum - right_sum)/(left_sum + right_sum+1))
    
    def forehead_sum(self) -> float:
        return float(np.sum(self.wrk*self.mask_forehead))
    
    def naso_tubularity(self) -> float:
        [x_min_r, x_max_r, y_min_r, y_max_r], [x_min_l, x_max_l, y_min_l, y_max_l]= get_naso_limits(coords=self.coords)
        wrk_l = self.wrk[y_min_l:y_max_l, x_min_l:x_max_l]
        wrk_r = self.wrk[y_min_r:y_max_r, x_min_r:x_max_r]
        img_l = self.img[y_min_l:y_max_l, x_min_l:x_max_l]
        img_r = self.img[y_min_r:y_max_r, x_min_r:x_max_r]
        mask_l = self.mask_naso_left[y_min_l:y_max_l, x_min_l:x_max_l]
        mask_r = self.mask_naso_right[y_min_r:y_max_r, x_min_r:x_max_r]

        self.tub_l, _, _ = FrangiFilter2D(cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY), np.array([1, 6]))*wrk_l*mask_l
        self.tub_r, _, _ = FrangiFilter2D(cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY), np.array([1, 6]))*wrk_r*mask_r

        return (np.sum(self.tub_l) + np.sum(self.tub_r))/(wrk_l.sum() + wrk_r.sum())
    
    def get_all_singleim_params(self):
        self.upper(), self.lower(), self.left(), self.right(), self.forehead(), self.naso_left(), self.naso_right()
        return self.overall_wrinkles(), self.left_to_right(), self.upper_to_lower(), self.forehead_sum(), self.naso_tubularity()

def jsi(src: np.ndarray, trg: np.ndarray):
    """Computes JSI of two numpy arrays"""
    inter = np.logical_and(src.astype(np.bool_), trg.astype(np.bool_)).sum()
    union = np.logical_or(src, trg).sum()

    return inter/union

def tubu_diff(face_src: Face, face_trg: Face, wrk_hat:np.ndarray, img_hat:np.ndarray):
    """
    Computes difference in tubularity between two images.
    :param face_src: Source face
    :param face_trg: Target image
    :param wrk_hat: Estimated aligned wrinkles
    :param img_aht: Estimated aligned image
    """
    tub_l_trg = face_trg.tub_l
    tub_r_trg = face_trg.tub_r

    [x_min_r, x_max_r, y_min_r, y_max_r], [x_min_l, x_max_l, y_min_l, y_max_l] = get_naso_limits(coords=face_trg.coords)
    wrk_l = wrk_hat[y_min_l:y_max_l, x_min_l:x_max_l]
    wrk_r = wrk_hat[y_min_r:y_max_r, x_min_r:x_max_r]
    img_l = img_hat[y_min_l:y_max_l, x_min_l:x_max_l]
    img_r = img_hat[y_min_r:y_max_r, x_min_r:x_max_r]
    mask_l = face_trg.mask_naso_left[y_min_l:y_max_l, x_min_l:x_max_l]
    mask_r = face_trg.mask_naso_right[y_min_r:y_max_r, x_min_r:x_max_r]

    tub_l_src, _, _ = FrangiFilter2D(cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY), np.array([1, 6]))*wrk_l*mask_l
    tub_r_src, _, _ = FrangiFilter2D(cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY), np.array([1, 6]))*wrk_r*mask_r
    return (np.abs(tub_l_trg - tub_l_src).sum()  + np.abs(tub_r_trg - tub_r_src).sum())/((wrk_l*mask_l).sum() + (wrk_r*mask_r).sum()+1)


def get_2img_params(face_src: Face, face_trg: Face, way = 2):
    """Computes all parameters for two images"""
    if way == 2:
        wrk_hat, _ = transf_source2target_2D(face_src.wrk.astype(np.float32), face_src.coords, face_trg.coords)
        img_hat, _ = transf_source2target_2D(face_src.img.astype(np.float32), face_src.coords, face_trg.coords)
    if way == 3:
        face_src.copmute_mesh()
        face_trg.copmute_mesh()
        valid_vert = face_src.mesh.surface_vertexes & face_trg.mesh.surface_vertexes
        wrk_hat = transf_source2target_piecwise(face_src.wrk.astype(np.float32), face_src.coords[valid_vert, :], face_trg.coords[valid_vert, :])
        img_hat = transf_source2target_piecwise(face_src.img.astype(np.float32), face_src.coords[valid_vert, :], face_trg.coords[valid_vert, :])
    
    JSI = jsi(wrk_hat, face_trg.wrk)
    JSI_U = jsi(wrk_hat * face_trg.mask_upper, face_trg.wrk * face_trg.mask_upper)
    JSI_D = jsi(wrk_hat * face_trg.mask_lower, face_trg.wrk * face_trg.mask_lower)
    JSI_L = jsi(wrk_hat * face_trg.mask_left, face_trg.wrk * face_trg.mask_left)
    JSI_R = jsi(wrk_hat * face_trg.mask_right, face_trg.wrk * face_trg.mask_right)
    TUBU_DIFF = tubu_diff(face_src, face_trg, wrk_hat, img_hat)
    return [JSI, JSI_U - JSI_D, JSI_L - JSI_R, TUBU_DIFF]
