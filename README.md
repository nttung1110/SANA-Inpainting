# SANA-Inpainting

This repository is my implementation attempt to enable image inpainting on [SANA](https://github.com/NVlabs/Sana)-a recently introduced text-to-image generation with high quality and highly efficient inference speed.

## Installation

Please refer to the original repository [SANA](https://github.com/NVlabs/Sana) for installation details.

## Inference
The main inference script that perform image inpainting for both SANA and SANA-Sprint are located in ```test_sana_inpaint.py``` and ```test_sana_sprint_inpaint.py```. 
Test samples which includes: edit_prompt, and inpainting mask are also provided in ```test_cases```

## Results
From left-to-right, source image and inpainted image result.
### SANA 40 steps
#### A basket of apples->A basket of durians
![A basket of apples->A basket of durians](relative/path/to/image.png)

![A vase of flowers->A vase of roses](relative/path/to/image.png)

### SANA-Sprint 10 steps
![A basket of apples->A basket of durians](relative/path/to/image.png)

![A vase of flowers->A vase of roses](relative/path/to/image.png)
## TODO
- [ ] Integrate as features into [SANA](https://github.com/NVlabs/Sana) repo

## Acknowledgement
If you are interested in this work, please refer to the original repository [SANA](https://github.com/NVlabs/Sana) and give them a star for their excellent SANA model. Also, your star for this repo would be really appreciated and encouraging for me. Thanks!
