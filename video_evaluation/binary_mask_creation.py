


import pygame
import numpy as np
from pygame.locals import *
from PIL import Image
import matplotlib.pyplot as plt
import copy

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(current_dir)

def get_circle_indices(N, M, radius, x, y):
    x_idx, y_idx = np.meshgrid(np.arange(M), np.arange(N))
    distances = np.sqrt((x_idx - x)**2 + (y_idx - y)**2)
    indices_in_circle = np.where(distances < np.sqrt(radius))
    return np.array(indices_in_circle).T

def merge_binary_mask(img: np.ndarray, original_image: np.ndarray, mask: np.ndarray, delete_mask: np.ndarray):
    img = img.astype(np.int32)
    img[mask] = [250, 0, 0]
    img = img.clip(0, 255)
    img = img.astype(np.uint8)
    img[delete_mask] = original_image[delete_mask]
    return img

def adjust_binary_mask(original_image: np.ndarray, binary_mask: np.ndarray = None):
    """
    A script in pygame to create mask. 
    Moving with pressed left button will draw the mask, moving with pressed right button will erase the mas
    Scroll to change brush size.
    Quit to return the mask.
    :param original_image: 
    """
    pygame.init()
    save_image = copy.deepcopy(original_image)
    height, width, c = original_image.shape
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Interactive Drawing")
    if binary_mask is None:
        binary_mask = np.zeros((original_image.shape[:2]), dtype=np.bool_)

    eraser_mask = np.zeros_like(binary_mask, dtype=np.bool_)
    image_surface = pygame.surfarray.make_surface(original_image.transpose((1, 0, 2)))

    drawing = False
    eraser = False
    radius = 3

    running = True
    counter = 15 # refresh rate
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    drawing = True
                elif event.button == 3:  # Right mouse button
                    drawing = True
                    eraser = True
                elif event.button == 4:  # Scroll Up
                    if radius < 50: radius += 1
                    print("Marker size: ", radius)
                    continue
                elif event.button == 5:  # Scroll Down
                    if radius > 1: radius -= 1
                    print("Marker size: ", radius)
                    continue
                x, y = event.pos
                circle = get_circle_indices(original_image.shape[0], original_image.shape[1], radius, x, y)
                if not eraser:
                    binary_mask[circle[:, 0], circle[:, 1]] = True
                else:
                    eraser_mask[circle[:, 0], circle[:, 1]]  = True
            elif event.type == MOUSEBUTTONUP:
                drawing = False
                eraser = False
            elif event.type == MOUSEMOTION and drawing:
                x, y = event.pos
                circle = get_circle_indices(original_image.shape[0], original_image.shape[1], radius, x, y)
                if not eraser:
                    binary_mask[circle[:, 0], circle[:, 1]] = True
                else:
                    eraser_mask[circle[:, 0], circle[:, 1]]  = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    x, y = pygame.mouse.get_pos()
                    drawing = True
                    eraser = True
                    circle = get_circle_indices(original_image.shape[0], original_image.shape[1], radius, x, y)
                    eraser_mask[circle[:, 0], circle[:, 1]]  = True                    
                    drawing = False
                    eraser = False
            
        binary_mask[eraser_mask] = 0
        if counter == 15:
            image_surface = pygame.surfarray.make_surface(merge_binary_mask(original_image, save_image, binary_mask, eraser_mask).transpose((1, 0, 2)))
            eraser_mask = np.zeros_like(original_image[:, :, 0], dtype=np.bool_)
            screen.blit(image_surface, (0, 0))
            pygame.display.flip()
            counter = 0
        else:
            counter += 1
    pygame.quit()
    return binary_mask

def show_wrinkle_mask(img, mask):
    img = copy.deepcopy(img)
    img[mask, :] = [255, 0, 0]
    plt.imshow(img)
    plt.show()

