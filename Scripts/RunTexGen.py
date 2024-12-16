from diffusers import StableDiffusionPipeline
import torch
import sys
import gc
print(torch.__version__)
model_id = "dream-textures/texture-diffusion"
project_path = "E:/LooseControlUE/LooseControlUE/"
prompt = "pbr brick wall"
neg_prompt = "(low quality, worst quality:1.3), lowres, signature, text, jpeg artifacts"
step = 30
seed = 42

print("in argument:", sys.argv[1:])

if len(sys.argv) >= 2:
    project_path = sys.argv[1]


if len(sys.argv) >= 4:
    prompt = sys.argv[2]
    neg_prompt = sys.argv[3]

if len(sys.argv) >= 5:
    step = int(sys.argv[4])

if len(sys.argv) >= 6:
    seed = int(sys.argv[5])

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
print("prompt:", prompt)
generator = torch.manual_seed(seed)
image = pipe(prompt, num_inference_steps=step, negative_prompt=neg_prompt, generator=generator).images[0]  

torch.cuda.empty_cache()
torch.clear_autocast_cache()

# sys.modules.pop('torch')
'''
返回rst.png是用于后面normal、displacement、roughness数据的处理
rst_albedo.png是用于光照贴图的处理
'''

# 需要upscale
# if upscale == True:    
#     # image = image.resize((256, 256))
#     from diffusers import StableDiffusionUpscalePipeline
#     # load model and scheduler
#     upscale_model_id = "stabilityai/stable-diffusion-x4-upscaler"
#     upscale_pipeline = StableDiffusionUpscalePipeline.from_pretrained(upscale_model_id, torch_dtype=torch.float16)
#     upscale_pipeline = upscale_pipeline.to("cuda")
#     # low vram
#     upscale_pipeline.enable_attention_slicing()
#     image_upscale = upscale_pipeline(prompt=prompt, negative_prompt=neg_prompt, image=image).images[0]
#     # 保存upscale版本    
#     image_upscale.save(project_path + "rst_albedo.png") # upscale版本
#     # 存一下Normal Gen使用的版本
#     if upscale_use_origin == True:    
#         image.save(project_path + "rst.png")   
#     else:
#         image_upscale.save(project_path + "rst.png")
    
# else:
# Albedo版本和normal gen用的是同一个版本    
image.save(project_path + "rst.png")    
image.save(project_path + "rst_albedo.png")

print(f"save to {project_path + 'rst.png'}")

# 清理缓存，否则有可能显存不被释放
# torch.cuda.empty_cache()
del torch
del pipe
gc.collect()