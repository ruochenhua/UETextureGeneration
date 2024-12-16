import sys
project_path = ""
if len(sys.argv) >= 2:
    project_path = sys.argv[1]
import os
import shutil
file_path = os.path.dirname(__file__)
shutil.copyfile(project_path + "rst.png", file_path + "/MMG/input/rst.png")


sys.path.append(file_path)
sys.path.append(file_path + "/MMG")
sys.path.append(file_path + "/MMG/utils")
print(file_path)

import cv2
import numpy as np
import torch
import MMG.utils.imgops as ops
import MMG.utils.architecture.architecture as arch

device = torch.device('cuda')
input_folder = os.path.normpath(file_path +"/MMG/input")
# output_folder = os.path.normpath(project_path +"MMG/output")
output_folder = os.path.normpath(project_path)

NORMAL_MAP_MODEL = file_path +'/MMG/utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
OTHER_MAP_MODEL = file_path +'/MMG/utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'

def process(img, model):
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output

def load_model(model_path):
    global device
    state_dict = torch.load(model_path)
    model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                            mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)

images=[]
for root, _, files in os.walk(input_folder):
    for file in sorted(files):
        if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga']:
            images.append(os.path.join(root, file))
print('input folder:', input_folder, len(images))
models = [
    # NORMAL MAP
    load_model(NORMAL_MAP_MODEL), 
    # ROUGHNESS/DISPLACEMENT MAPS
    load_model(OTHER_MAP_MODEL)
    ]
for idx, path in enumerate(images, 1):
    base = os.path.splitext(os.path.relpath(path, input_folder))[0]
    output_dir = os.path.dirname(os.path.join(output_folder, base))
    os.makedirs(output_dir, exist_ok=True)
    print(idx, base)
    # read image
    try: 
        img = cv2.imread(path, cv2.cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
    # Seamless modes
    img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_WRAP)
    # if args.seamless:
    #     img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
    # elif args.mirror:
    #     img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    # elif args.replicate:
    #     img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

    img_height, img_width = img.shape[:2]
    print("img:",img_height,img_width)
    tile_size = 512
    # Whether or not to perform the split/merge action
    do_split = img_height > tile_size or img_width > tile_size
    # do_split = False

    if do_split:
        rlts = ops.esrgan_launcher_split_merge(img, process, models, scale_factor=1, tile_size=tile_size)
    else:
        rlts = [process(img, model) for model in models]

    # if args.seamless or args.mirror or args.replicate:
    #     rlts = [ops.crop_seamless(rlt) for rlt in rlts]

    normal_map = rlts[0]
    roughness = rlts[1][:, :, 1]
    displacement = rlts[1][:, :, 0]

    # if args.ishiiruka_texture_encoder:
    #     r = 255 - roughness
    #     g = normal_map[:, :, 1]
    #     b = displacement
    #     a = normal_map[:, :, 2]
    #     output = cv2.merge((b, g, r, a))
    #     cv2.imwrite(os.path.join(output_folder, '{:s}.mat.png'.format(base)), output)
    # else:
    # r = normal_map[:,:,0]
    # g = normal_map[:,:,1]
    # b = normal_map[:,:,2]
    # print("type :", type(b))
    
    # output_normal = cv2.merge((r,g,b))
    
    normal_name = '{:s}_Normal.png'.format(base)
    cv2.imwrite(os.path.join(output_folder, normal_name), normal_map)

    rough_name =  '{:s}_Roughness.png'.format(base)
    rough_img = roughness
    cv2.imwrite(os.path.join(output_folder, rough_name), rough_img)

    displ_name =  '{:s}_Displacement.png'.format(base)
    cv2.imwrite(os.path.join(output_folder, displ_name), displacement)

torch.cuda.empty_cache( )
# sys.modules.pop('torch')