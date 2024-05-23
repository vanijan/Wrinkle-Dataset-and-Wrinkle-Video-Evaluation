__author__ = 'solivr'
# a script for wrinkle evaluations of videos

import numpy as np
import os
from segmentation import segment
import monai
import torch
import cv2
from face import Face, get_2img_params, resize_coords
device = "cpu" # adjust to your device
DIR = ""

def load_model(model_type: torch.nn.Module, directory: str = DIR, model_path = "model.pth"):
  model_type.load_state_dict(torch.load(os.path.join(directory, model_path)))


video_folder = "" #...............
HEIGHT = 640
WIDTH = 480
model = monai.networks.nets.UNet(2, 3, 1, [32, 64, 128, 256, 512], [2, 2, 2, 2], num_res_units=4).to(device)
load_model(model, "" , "") #(model, directory, filename)

for file in os.listdir(video_folder):
    with torch.no_grad():
        video_capture = cv2.VideoCapture(os.path.join(video_folder, file))

        quit = False
        batchsize = 5
        batch = torch.zeros((batchsize, 3, HEIGHT, WIDTH)) #prealocate memory for the batch
        coords_batch = []
        params_singleim = []
        diff_img_params = []
        diff_img_5_params = []

        while True:
            try: # try-except block for no face detection error
                for i in range(batchsize):
                    ret, frame = video_capture.read()  # Read a frame
                    if not ret:
                        quit = True
                        break  # Break the loop if there are no more frames
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img, _, coords, _, _ = segment(img)
                    orig_shape = img.shape
                    coords = resize_coords(orig_shape[:2], (HEIGHT, WIDTH), coords) #resize coordinates
                    coords_batch.append(coords) #save coordinates
                    img = cv2.resize(img, (WIDTH, HEIGHT)).astype(np.float32)/255 #resize the image
                    img_t = torch.tensor(img.transpose(2, 0, 1)) 
                    batch[i, ...] = img_t #save the image to the batch
            except:
                continue
            if quit:
                print(f"ended {file}") 
                break
            preds = torch.nn.functional.sigmoid(model(batch.to(device))).to("cpu")
            batch = batch.to("cpu")
            faces = []
            for i in range(batchsize):
                wrk = np.array(preds[i, 0, :, :]) > 0.5
                img = np.array(batch[i, ...]).transpose(1, 2, 0)
                faces.append(Face(img, wrk, coords_batch[i], copmute_mesh=False))
            
            for (i, face) in enumerate(faces):
                print("here")
                face: Face
                #face.show_img_with_wrinkles() #visualize detection
                params_singleim.append(face.get_all_singleim_params())
                if i > 0:
                    diff_img_params.append(get_2img_params(faces[i], faces[i-1]))
            diff_img_5_params.append(get_2img_params(faces[0], faces[-1]))

            #SAVE PARAMETERS... #np.save(np.arrray(diff_img_params))
        print("finished ok")

        
        
        
        
    