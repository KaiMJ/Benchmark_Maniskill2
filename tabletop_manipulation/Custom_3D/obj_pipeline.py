from diffusers import DiffusionPipeline
import torch
from tabletop_manipulation.Custom_3D.convert_color import convert
from utils.run import process_images
from IPython.display import display  
import matplotlib.pyplot as plt
from glob import glob
import os

# load both base & refiner
# 7.3GB
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda:1")

# 4.4 GB
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

refiner.to("cuda:1")
print("Base and refiner loaded")


prompt = "zebra"
name = prompt


# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompts = [prompt + " with a transparent background"] * 9

# run both experts
image = base(
    prompt=prompts,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

image = refiner(
    prompt=prompts,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images


fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.ravel()

for idx, img in enumerate(image):
    axs[idx].imshow(img)
    axs[idx].axis('off')
    axs[idx].set_title(f"{idx}")

plt.show()


image[8].save(f"images/{name}.png")


process_images([f"images/{name}.png"], output_dir=f"models/{name}")