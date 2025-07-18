import torch
import json
import time
import numpy as np
from src.pipeline_sana_sprint_inpaint import SanaSprintInpaintingPipeline
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def process_input_mask(mask_path, target_size=1024):
    mask_img = np.array(Image.open(mask_path).resize((target_size, target_size)).convert("RGB"))
    mask_img = np.sum(mask_img, axis=2)
    mask_img = np.where(mask_img!=0, 1, 0)
    
    return mask_img

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

    sana = SanaSprintInpaintingPipeline("./configs/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml")
    sana.from_pretrained("hf://Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth")

    with open("./test_cases/test.json", "r") as fp:
        data = json.load(fp)

    for sample in data:
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]
        src_p = sample["src_prompt"]
        edit_p = sample["edit_prompt"]

        # process source image for visualize
        src_img =  Image.open(img_path).convert("RGB").resize(
            (1024, 1024), Image.Resampling.LANCZOS
        )
        src_img = transforms.ToTensor()(
            src_img
        ) * 2 - 1
        src_img = src_img.to("cuda:0").unsqueeze(0)

        # process mask for blending
        mask_image = process_input_mask(mask_path)
        # mask_image = get_dilated_mask(
        #     mask_image,
        #     dilation_size=3,
        # )

        src_img_with_mask = overlay_mask_on_image(src_img, mask_image)

        vis_results = []

        start_time = time.time()
        # Inpaint image       
        inpaint_image = sana(
            img_path=img_path,
            edit_prompt=edit_p,
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=10,
            mask_image=mask_image,
            generator=generator,
        )
        
        end_time = time.time()
        print(f"Finish inpainting within {end_time-start_time}")

        vis_results = torch.cat([src_img_with_mask, inpaint_image])
        
        save_image(vis_results, f'./results_inpaint_sana_sprint/{src_p}->{edit_p}.png', normalize=True, value_range=(-1, 1))

