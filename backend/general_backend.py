import os
import torch
import re
import cv2
import subprocess
from PIL import Image
from IPython.display import display
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import numpy as np

# Attempt to import basicsr and gfpgan
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from gfpgan import GFPGANer
    gfpgan_available = True
except ImportError as e:
    print(f"Failed to import basicsr or gfpgan: {e}")
    print("GFPGAN face enhancement will be disabled.")
    gfpgan_available = False
    RRDBNet = None
    GFPGANer = None

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Download GFPGAN weights
def download_gfpgan_weights():
    if not gfpgan_available:
        return None
    model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    model_dir = "gfpgan/weights"
    model_path = os.path.join(model_dir, "GFPGANv1.4.pth")
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        print("Downloading GFPGAN model weights...")
        subprocess.run(["wget", model_url, "-O", model_path], check=True)
    return model_path

# Initialize GFPGAN
face_enhancer = None
if gfpgan_available:
    try:
        model_path = download_gfpgan_weights()
        if model_path:
            face_enhancer = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=device
            )
    except Exception as e:
        print(f"Error initializing GFPGAN: {e}")
        face_enhancer = None

# Style detection
def interpret_prompt(prompt):
    styles = {
        "anime": ["anime", "manga", "japanese"],
        "ghibli": ["ghibli", "studio ghibli", "my neighbor totoro"],
        "realism": ["realistic", "photorealistic", "detailed"],
        "cyberpunk": ["cyberpunk", "futuristic", "neon city"],
        "watercolor": ["watercolor", "painting", "soft colors"],
        "oil painting": ["oil painting", "classic art", "canvas"],
        "pixel art": ["pixel art", "8-bit", "retro"],
        "cartoon": ["cartoon", "comic", "animated"]
    }
    for style, keywords in styles.items():
        if any(re.search(rf"\b{kw}\b", prompt, re.IGNORECASE) for kw in keywords):
            print(f"Suggested style: {style}")
            return style
    print("No specific style detected. Using 'default'.")
    return "default"

# Stable Diffusion pipeline loader
def get_pipeline(style):
    model_ids = {
        "anime": "Linaqruf/animagine-xl",
        "ghibli": "nitrosocke/Ghibli-Diffusion",
        "realism": "stabilityai/stable-diffusion-2",
        "cyberpunk": "nitrosocke/Cyberpunk-Diffusion",
        "watercolor": "nitrosocke/Watercolor-Diffusion",
        "oil painting": "nitrosocke/Oil-Painting-Diffusion",
        "pixel art": "nitrosocke/Pixel-Art-Diffusion",
        "cartoon": "nitrosocke/Cartoon-Diffusion"
    }
    model_id = model_ids.get(style, "runwayml/stable-diffusion-v1-5")

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    pipeline.enable_attention_slicing()
    pipeline.safety_checker = None

    return pipeline

# Image generation with face enhancement
def generate_images(prompt, n_images=4, style="default"):
    pipeline = get_pipeline(style)
    with torch.inference_mode():
        images = pipeline(
            prompt,
            num_images_per_prompt=n_images,
            guidance_scale=7.5,
            height=512,
            width=512,
            num_inference_steps=30
        ).images

    enhanced_images = []
    for img in images:
        if face_enhancer and gfpgan_available:
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            try:
                _, _, restored_face = face_enhancer.enhance(
                    img_np, has_aligned=False, only_center_face=False, paste_back=True
                )
                enhanced_images.append(Image.fromarray(cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)))
            except Exception as e:
                print(f"Error enhancing image: {e}")
                enhanced_images.append(img)
        else:
            enhanced_images.append(img)

    return enhanced_images

# Image grid for display
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# ControlNet modification function
def controlnet_generate(image_path, prompt, model_type):
    controlnet_models = {
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "depth": "lllyasviel/control_v11f1p_sd15_depth",
        "pose": "lllyasviel/control_v11p_sd15_openpose",
        "scribble": "lllyasviel/control_v11p_sd15_scribble",
        "segmentation": "lllyasviel/control_v11p_sd15_seg"
    }
    if model_type not in controlnet_models:
        raise ValueError("Choose from: canny, depth, pose, scribble, segmentation")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_models[model_type],
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    control_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    control_pipeline.enable_attention_slicing()

    input_image = Image.open(image_path).convert("RGB").resize((512, 512))
    with torch.inference_mode():
        result = control_pipeline(
            prompt=prompt,
            image=input_image,
            guidance_scale=7.5,
            num_inference_steps=30
        ).images[0]

    return result

# User Interaction
mode = input("Generate a new image (1) or modify existing image (2)? Enter 1 or 2: ")

if mode == "1":
    prompt = input("Enter your image prompt: ")
    suggested_style = interpret_prompt(prompt)
    style = input(f"Choose style ({suggested_style} recommended, or leave empty): ") or suggested_style
    images = generate_images(prompt, n_images=4, style=style)
    grid = image_grid(images, 2, 2)
    display(grid)

elif mode == "2":
    image_path = input("Enter path to your image: ")
    prompt = input("Enter your modification prompt: ")
    model_type = input("Choose ControlNet model (canny/depth/pose/scribble/segmentation): ")
    generated_image = controlnet_generate(image_path, prompt, model_type)
    display(generated_image)

else:
    print("Invalid input! Please enter 1 or 2.")