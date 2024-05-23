# Public Wrinkle Dataset and Wrinkle Video Evaluation

This repository provides a manually annotated dataset of 674 wrinkle files in in .bmp format. The files are located in the wrinkles folder.
The original images come from the FFHQ dataset which can be found here: https://github.com/NVlabs/ffhq-dataset, their numbers were preserved.
The dataset contains some harder images with glasses, faces not shot from completely frontal view, and harsher light conditions.
The original images have to be cropped for wrinkles to match them, information about the crop limits can be found in the file wrinkles/crop_limits.npy.

The wrinkles were labeled by a technical university student, not a dermatologist, and we do not provide a warranty of any kind.
However, we hope for this dataset to ease the training or evaluation of wrinkle segmentation models. If you find this dataset helpful, 
please cite the diploma thesis "Automatic Analysis of Facial Wrinkle Characteristics in People with Parkinson's Disease".

crop_limits.npy are a saved (674, 5) numpy array where one row corresponds to [image_number, min_h, max_h, min_w, max_w].

## Wrinkle Video Evaluation
This repository also contains files in the video_evaluation folder used for video evaluation in the thesis and is provided for reproducibility.
