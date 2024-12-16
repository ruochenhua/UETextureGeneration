# 漫反射贴图生成：
调研模型：texture diffusion（https://huggingface.co/dream-textures/texture-diffusion）
模型基于stable diffusion 2 base，通过DreamBooth微调，可通过文生图的方式生成材质的漫反射贴图，尽量不包含光照和阴影信息

# 法线贴图生成：
为了有更加真实的光照表现效果，贴图一般会配合法线贴图使用。
从漫反射贴图生成法线贴图的库有好几个，比如说deepbump（https://github.com/HugoTini/DeepBump），demo使用的是Material-Map-Generator(https://github.com/joeyballentine/Material-Map-Generator)，因为它还可以生成DisplacementMap和RoughnessMap，这两张贴图同样可以增强模型在3D光照环境下的表现。

## 置换贴图：
上文提到了置换贴图（DisplacementMap），这个资源也是一个提升模型显示效果的手段。法线贴图增加模型细节是不需要修改模型本身的顶点形状的，只是通过提供更为细致的平面法线信息辅助光照计算。
置换贴图则是可以真实的修改模型的形状。
在UE中可以使用模型工具通过DisplacementMap来丰富模型的细节，也有另外一种方法“视察遮挡映射（ParallaxOcclusionMapping)”，这里我在UE中使用这种方法。具体细节就不在本文档阐述了，下面是对比效果。
首先是没有使用DisplacementMap的情况，可以看到由于模型本身的形状在只是NormalMap的影响下是没有变化的，所以白色球体的阴影在不平整的墙面上却是平整的。

# Upscale：
使用Upscale Pipeline stable-diffusion-x4-upscaler-img2img（https://huggingface.co/radames/stable-diffusion-x4-upscaler-img2img），将原有的贴图分辨率从512X512提升到2048X2048，模型精度有比较大的提升，但是显存需求显著增大，并且消耗时间显著增长。

# 安装步骤：
  1. 找到引擎的python地址，如F:\UnrealEngine\UE_5.3\Engine\Binaries\ThirdParty\Python3\Win64，以这个Python的路径，带入cmd，运行pip安装对应的库（如下面安装numpy）
  2. 需要安装的库：
    transformers, diffusers, accelerate（hugging face）
    pytorch（https://pytorch.org/get-started/locally/， https://pytorch.org/get-started/previous-versions/)
    python的版本可能不支持最新的pytorch版本，如ue5.3使用的python 3.9.7只能支持到pytorch2.1，需要根据版本来安装合适的版本
    numpy == 1.24.1
    opencv pip install opencv-python==4.6.0.66
    
