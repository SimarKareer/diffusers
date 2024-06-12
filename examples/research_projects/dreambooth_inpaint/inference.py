import PIL
import requests
import torch
import torchvision.transforms.functional as TVF
import torchvision
from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline

import glob

import os
import argparse

def color_map(image, orig_pixel, new_pixel):
    """
    image: (C, H, W)
    orig_pixel: torch.tensor([r, g, b])
    new_pixel: torch.tensor([r, g, b])
    map all orig_pixel to new_pixel in image
    """

    mask = torch.all(image == orig_pixel[:, None, None], dim=0)
    image[:, mask] = new_pixel[:, None]
    return image

def dilate(mask_image):
    kernel = torch.tensor([ [1., 1, 1],
                        [1, 1, 1],
                        [1, 1, 1] ]).cuda()
    kernel = kernel[None, None, :, :]
    for i in range(10):
        mask_image = torch.clamp(torch.nn.functional.conv2d(mask_image, kernel, padding=(1, 1)), 0, 1)

    return mask_image


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def inference_image(pipe, img_path, mask_path, prompt):
    #set torch seed
    torch.manual_seed(0)

    # img_path = "/nethome/skareer6/flash9/Projects/EgoPlay/diffusers/data/hand_images/demo_3.png"
    # mask_path = "/nethome/skareer6/flash9/Projects/EgoPlay/diffusers/data/hand_images/mask_demo_3.png"

    init_image = torchvision.io.read_image(img_path).to("cuda")[:3]
    mask_image = torchvision.io.read_image(mask_path).to("cuda")[[0]]
    init_image = TVF.resize(init_image, (512, 512))
    mask_image = TVF.resize(mask_image, (512, 512))



    # # # # in mask image change all instances of black to white and all instances of white to black
    # mask_image[mask_image == 0] = 254
    # mask_image[mask_image == 255] = 0
    # mask_image[mask_image == 254] = 255

    # init_image = color_map(init_image, torch.tensor([0, 0, 0]).cuda().to(torch.uint8), torch.tensor([255, 0, 0]).cuda().to(torch.uint8))
    # breakpoint()

    # mask_image = mask_image.unsqueeze(0)
    # init_image = init_image.unsqueeze(0)
    mask_image = mask_image / 255
    mask_image = dilate(mask_image)

    #apply dilution to mask image

    init_image = init_image / 255.0


    # # set mask_image to all 0 with a box of 1 in the center
    # mask_image = torch.zeros_like(mask_image)
    # mask_image[:, 256-128:256+128, 256-128:256+128] = 1.0
    # save mask_image for visualization


    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    # save pil image
    # image.save("inpainting.jpg")
    return image, mask_image


def inference_folder(folder, ckpt, prompt, output_folder="outputs"):
    """
    folder: run inference on all .png in the folder
    """
    # images = sorted(glob.glob("/nethome/skareer6/flash9/Projects/EgoPlay/diffusers/data/hand_images/*.png"))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ckpt, torch_dtype=torch.float16, safety_checker = None,
    )
    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker = None,
    # )
    pipe = pipe.to("cuda")
    # output_folder = os.path.join("slurm", output_folder)
    output_folder = os.path.join(output_folder, "outputs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("MAKING OUTPUT FOLDER", output_folder)

    for i in range(len(os.listdir(folder))):
        img_path = os.path.join(folder, f"demo_{i}.png")
        mask_path = os.path.join(folder, f"mask_demo_{i}.png")
        img, mask_image = inference_image(pipe, img_path, mask_path, prompt)
        img.save(os.path.join(output_folder, f"inpainting_{i}.jpg"))
        torchvision.io.write_png(mask_image.cpu().to(torch.uint8)*255, "mask_image_{i}.png")
        

def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="ckpt to eval")
    parser.add_argument("--output-folder", type=str, help="output-folder")
    parser.add_argument("--prompt", type=str, help="prompt")
    args = parser.parse_args()

    print("got name", args.output_folder, "ckpt", args.ckpt)

    inference_folder("/nethome/skareer6/flash9/Projects/EgoPlay/diffusers/data/hand_images", ckpt=args.ckpt, prompt=args.prompt, output_folder=args.output_folder)

main()