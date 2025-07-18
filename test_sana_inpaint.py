import torch
import json
import time
import cv2
import numpy as np

from src.pipeline_sana_inpaint import SanaPipelineInpainting
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image

def process_input_mask(mask_path, target_size=1024):
    mask_img = np.array(Image.open(mask_path).resize((target_size, target_size)).convert("RGB"))
    mask_img = np.sum(mask_img, axis=2)
    mask_img = np.where(mask_img!=0, 1, 0)
    
    return mask_img

def get_dilated_mask(mask, dilation_size=1, erosion_shape=cv2.MORPH_RECT):

    element = cv2.getStructuringElement(
        erosion_shape,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size),
    )

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    dilation_mask = cv2.dilate(mask * 255 / 255, element)

    return dilation_mask

def overlay_mask_on_image(
        image_tensor, mask_np,
        alpha=0.5, overlay_color=(1, 0, 0),
        device="cuda"
    ):
    mask_tensor = torch.from_numpy(mask_np).float().to(device)  # (H, W)
    if mask_tensor.max() > 1:
        mask_tensor /= 255.0
    mask_tensor = mask_tensor.clamp(0, 1)

    # Convert overlay color to a tensor and expand
    color_tensor = torch.tensor(overlay_color).view(3, 1, 1).to(device)

    # Blend the overlay on top of the original image
    overlay = image_tensor * (1 - alpha * mask_tensor) + color_tensor * (alpha * mask_tensor)

    return overlay


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(42)
    NUM_STEP = 40
    negative_prompt = "bad quality"

    model = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"

    pipe = SanaPipelineInpainting.from_pretrained(
        model,
        torch_dtype=torch.bfloat16, 
        device=device,
    ).to(device)
    
    with open("./test_cases/test.json", "r") as fp:
        data = json.load(fp)


    for sample in data:
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]
        src_p = sample["src_prompt"]
        edit_p = sample["edit_prompt"]

        # Read source image for inpainting
        src_img =  Image.open(img_path).convert("RGB").resize(
            (1024, 1024), Image.Resampling.LANCZOS
        )

        # Process source image for visualize
        vis_src_img = transforms.ToTensor()(
            src_img
        ).unsqueeze(dim=0).to("cuda:0")

        # process mask for blending
        mask = process_input_mask(mask_path)
        mask = get_dilated_mask(
            mask,
            dilation_size=10,
        )

        vis_src_img_with_mask = overlay_mask_on_image(vis_src_img, mask)

        start_time = time.time()

        # Inpaint image
        inpaint_image = pipe(
            edit_p,
            generator=generator,
            src_image=src_img,
            mask=mask,
            height=1024,
            width=1024,
            num_inference_steps=NUM_STEP,
            negative_prompt=negative_prompt,
        ).images[0]

        inpaint_image = ToTensor()(
            inpaint_image
        ).unsqueeze(dim=0).to("cuda:0")

        end_time = time.time()
        print(f"Finish editing within {end_time-start_time}")

        # vis_results = torch.cat([src_img, rec_image, inpaint_image])
        vis_results = torch.cat([vis_src_img_with_mask, inpaint_image])
        
        save_image(vis_results, f'./results_inpaint_sana/{src_p}->{edit_p}.png')
