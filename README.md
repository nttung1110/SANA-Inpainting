# SANA-Inpainting

This repository implements image inpainting on [SANA](https://github.com/NVlabs/Sana), a recently introduced text-to-image generation with high quality and highly efficient inference speed.

## Installation

Please refer to the original repository [SANA](https://github.com/NVlabs/Sana) for installation details.

## Inference
Please refer to inference scripts that perform image inpainting for both SANA and SANA-Sprint at ```test_sana_inpaint.py``` and ```test_sana_sprint_inpaint.py```. 
Test samples are also provided with edit_prompt, and inpainting mask in ```test_cases```.

## Results
From left-to-right, source image and inpainted image result.
### SANA 40 steps
#### A basket of apples->A basket of durians
![A basket of apples->A basket of durians](results_inpaint_sana/A%20basket%20of%20apples-%3EA%20basket%20of%20durians.png)

![A vase of flowers->A vase of roses](results_inpaint_sana/A%20vase%20of%20flowers-%3EA%20vase%20of%20roses.png)

### SANA-Sprint 10 steps

![A basket of apples->A basket of durians](results_inpaint_sana_sprint/A%20basket%20of%20apples-%3EA%20basket%20of%20durians.png)

![A vase of flowers->A vase of roses](results_inpaint_sana_sprint/A%20vase%20of%20flowers-%3EA%20vase%20of%20roses.png)
## TODO
- [ ] Integrate as features into [SANA](https://github.com/NVlabs/Sana) repo.

## Acknowledgement
If you are interested in this work, please refer to the original repository [SANA](https://github.com/NVlabs/Sana) and give them a star for their excellent SANA model. Also, your star for this repo would be really appreciated and encouraging for me. Thanks!
