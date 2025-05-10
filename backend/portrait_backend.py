import os
import torch
import re
import cv2
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from huggingface_hub import login

# Hugging Face Authentication
def authenticate_huggingface():
    try:
        hf_token = os.getenv("HF_TOKEN") or input("Enter your Hugging Face token (or press Enter to skip): ")
        if hf_token:
            login(token=hf_token)
            print("Authenticated with Hugging Face.")
        else:
            print("No token provided. Running without authentication.")
    except Exception as e:
        print(f"Authentication failed: {e}")

authenticate_huggingface()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Prompt interpretation for style suggestion
def interpret_prompt(prompt):
    styles = {
        "anime": ["anime", "manga", "japanese"],
        "ghibli": ["ghibli", "studio ghibli"],
        "realism": ["realistic", "photorealistic", "portrait"],
        "cyberpunk": ["cyberpunk", "futuristic", "neon"],
        "watercolor": ["watercolor", "painting"],
        "oil painting": ["oil painting", "classic art"],
        "pixel art": ["pixel art", "8-bit"],
        "cartoon": ["cartoon", "comic"]
    }
    for style, keywords in styles.items():
        if any(re.search(rf"\b{kw}\b", prompt, re.IGNORECASE) for kw in keywords):
            print(f"Detected style: {style}")
            return style
    print("Defaulting to 'realism' style.")
    return "realism"

# Enhance image with CLAHE and Unsharp Mask
def enhance_image(image):
    try:
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        result_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
        return result_img.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=1))
    except Exception as e:
        print(f"Enhancement error: {e}")
        return image

# Upscale image by 2x
def upscale_image(image):
    try:
        return image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    except Exception as e:
        print(f"Upscaling error: {e}")
        return image

# Load appropriate model
def get_pipeline(style):
    model_ids = {
        "anime": "Linaqruf/animagine-xl",
        "ghibli": "nitrosocke/Ghibli-Diffusion",
        "realism": "runwayml/stable-diffusion-v1-5",
        "cyberpunk": "nitrosocke/Cyberpunk-Diffusion",
        "watercolor": "nitrosocke/Watercolor-Diffusion",
        "oil painting": "nitrosocke/Oil-Painting-Diffusion",
        "pixel art": "nitrosocke/Pixel-Art-Diffusion",
        "cartoon": "nitrosocke/Cartoon-Diffusion"
    }
    model_id = model_ids.get(style, "runwayml/stable-diffusion-v1-5")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    pipeline.enable_attention_slicing()
    pipeline.safety_checker = None
    return pipeline

# Generate new images
def generate_images(prompt, n_images=4, style="realism"):
    pipeline = get_pipeline(style)
    negative_prompt = (
        "deformed iris, deformed pupils, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, "
        "extra fingers, mutated hands, poorly drawn face, mutation, deformed, blurry, bad anatomy"
    )
    with torch.inference_mode():
        output = pipeline(prompt, num_images_per_prompt=n_images, guidance_scale=6.5,
                          height=768, width=768, num_inference_steps=40,
                          negative_prompt=negative_prompt)
    images = output.images
    return [upscale_image(enhance_image(img)) for img in images]

# ControlNet generation
def controlnet_generate(image_path, prompt, model_type):
    controlnet_models = {
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "depth": "lllyasviel/control_v11f1p_sd15_depth",
        "pose": "lllyasviel/control_v11p_sd15_openpose",
        "scribble": "lllyasviel/control_v11p_sd15_scribble",
        "segmentation": "lllyasviel/control_v11p_sd15_seg"
    }

    if model_type not in controlnet_models:
        raise ValueError("Invalid model_type. Choose from: canny, depth, pose, scribble, segmentation.")

    controlnet = ControlNetModel.from_pretrained(controlnet_models[model_type], torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    pipeline.enable_attention_slicing()
    pipeline.safety_checker = None

    image = Image.open(image_path).convert("RGB").resize((512, 512))

    with torch.inference_mode():
        result = pipeline(prompt=prompt, image=image, num_inference_steps=40,
                          guidance_scale=6.5, negative_prompt="deformed, blurry, bad anatomy").images[0]
    return upscale_image(enhance_image(result))

# Create grid for display
def image_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        grid.paste(img, box=(idx % cols * w, idx // cols * h))
    return grid

# Main loop
if __name__ == "__main__":
    choice = input("Generate new (1) or modify with ControlNet (2)? ")
    if choice == "1":
        prompt = input("Enter image prompt: ")
        style = interpret_prompt(prompt)
        custom_style = input(f"Enter style or leave blank for '{style}': ") or style
        imgs = generate_images(prompt, style=custom_style)
        if imgs:
            grid = image_grid(imgs, 2, 2)
            display(grid)
    elif choice == "2":
        img_path = input("Enter path to base image: ")
        prompt = input("Enter modification prompt: ")
        model = input("ControlNet type (canny/depth/pose/scribble/segmentation): ")
        result = controlnet_generate(img_path, prompt, model)
        if result:
            display(result)
    else:
        print("Invalid choice.")
