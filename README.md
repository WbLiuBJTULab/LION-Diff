# LION-Diff: Latent Diffusion-Based Voxel Enhancement \\ for Lidar-based 3D Object Detection


[Project]() | [Paper]() 

Implementation of LION-Diff for lidar-based 3D object detection task

Our project is based on [LION](https://github.com/happinesslz/LION) 


## Requirements:

Make sure the following environments are installed.

...

The code was tested on Unbuntu with GTX 4090ti. 

## Main change

Our project integrate diffusion-model into the OpenPCD detection framework, 
the modification location is as follows: 

```text
LION-Diff/
└── pcdet/
    └── models/
        ├── backbones_3d/
        │   └── lion-diff_backbone_one_stride.py 
        └── model_utils/
            └── diffusion_modules/
                ├── diffusion_utils.py
                ├── prepare_diffusion.py
                ├── radar_cond_diff_denoise.py
                └── unet.py  

```



## Data



```bash

```

## Pretrained models


## Training:

```bash

```

Please refer to the python file for optimal training parameters.

## Testing:

```bash

```

## Results

## Reference


## Acknowledgement

We thank these great works: 
[LION](https://github.com/happinesslz/LION) 