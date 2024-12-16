# image = image.resize((256, 256))
from diffusers import StableDiffusionUpscalePipeline
import sys
import gc
import torch

from PIL import Image

print("in argument:", sys.argv[1:])
project_path = "E:/LooseControlUE/LooseControlUE/"
is_4X_upscale = True

prompt = "pbr brick wall"
neg_prompt = "(low quality, worst quality:1.3), lowres, signature, text, jpeg artifacts"

if len(sys.argv) >= 5:
    project_path = sys.argv[1]
    prompt = sys.argv[2]
    prompt = sys.argv[3]
    is_4X_upscale = sys.argv[4] == 'true'

# load model and scheduler
upscale_model_id = "stabilityai/stable-diffusion-x4-upscaler"
upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(upscale_model_id, torch_dtype=torch.float16)
upscale_pipeline = upscale_pipeline.to("cuda")
# low vram
upscale_pipeline.enable_attention_slicing()

origin_image = Image.open(project_path + "rst_albedo.png")
if is_4X_upscale == False:
    origin_image.resize((256, 256))
    print("do 2x scale")
else:
    print("do 4x scale")

image_upscale = upscale_pipeline(prompt=prompt, negative_prompt=neg_prompt, image=origin_image).images[0]
# 保存upscale版本    
image_upscale.save(project_path + "rst_albedo.png") # upscale版本

torch.cuda.empty_cache()
torch.clear_autocast_cache()
# pop掉对torch的引用
# sys.modules.pop('torch')
del torch
del upscale_pipeline
gc.collect()