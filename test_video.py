import argparse
from functions import inverse_normalize
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
import torchvision
from torchvision.transforms.functional import center_crop
from torchvision.transforms.transforms import CenterCrop, RandomCrop

from torchvision.utils import save_image
from pathlib import Path
import models.modeling as modeling

import time
import numpy as np
import random
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    #transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    #transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform_list)
    return transform

# def style_transform(h,w):
#     k = (h,w)
#     size = int(np.max(k))
#     size = int(size//4*4)
#     #print(type(size))
#     transform_list = []
     
#     transform_list.append(transforms.Resize(size))
    
#     #transform_list.append(transforms.CenterCrop((h,w)))
#     transform_list.append(transforms.ToTensor())
#     transform = transforms.Compose(transform_list)
#     return transform
localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default='input/content/golden_gate.jpg',
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='data/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default='input/style/la_muse.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default='data/style',
                    help='Directory path to a batch of style images')
parser.add_argument('--decoder', type=str, default='experiments/decoder2.pth.tar')

# Additional options
parser.add_argument('--size', type=int, default=256,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='video_images',
                    help='Directory to save the output image(s)')


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
assert (args.style or args.style_dir)

content_dir = Path(args.content_dir)
content_paths = [f for f in content_dir.glob('*')]
style_dir = Path(args.style_dir)
style_paths = [f for f in style_dir.glob('*')]

model = modeling.VisionTransformer(modeling.CONFIGS['ViT-B_16'])
#model = modeling.VisionTransformer(modeling.CONFIGS['my_Vit'])
model.transformer.load_state_dict(torch.load('./ckpts/exp9_random_ViT_wct/vit_iter_225000.pth'))
#model.load_from(np.load('vit_checkpoints/imagenet21k_ViT-B_16.npz'))
model.transformerdecoder.load_state_dict(torch.load('./ckpts/exp9_random_ViT_wct/decoder_iter_225000.pth'))
model.to(device)
model.eval()
print("load done!")
from math import ceil
#-----------------------start------------------------

# for content_path in content_paths:
#     for style_path in style_paths:
#         with torch.no_grad():

#             content = Image.open(str(content_path)).convert('RGB')
#             img_transform = content_transform()

#             content = img_transform(content)
#             #h, w, c = np.shape(content)
#             content = content.to(device).unsqueeze(0)
#             style = Image.open(str(style_path)).convert('RGB')
#             img_transform = style_transform()
#             style = img_transform(style).to(device).unsqueeze(0)
            
#             b, c, h, w = content.size()
#             hn = ceil(h/224)
#             wn = ceil(w/224)
#             output = torch.zeros_like(content)
#             for i in range(hn):
#                 for j in range(wn):
#                     flag = False
#                     x = i * 224
#                     y = j * 224
#                     temp = content[:,:,x:x+224,y:y+224].clone().detach()
#                     if (x + 224) > h or (y + 224) > w:
#                         temp = content[:,:,x:min(x+224,h),y:min(y+224,w)]
#                         #temp= nn.ZeroPad2d([0,max(0,224-w+y),0,max(224+x-h,0)])(temp)
#                         #temp= nn.ConstantPad2d([0,max(0,224-w+y),0,max(224+x-h,0)],0)(temp)
#                         temp = nn.ReplicationPad2d([0,max(0,224-w+y),0,max(224+x-h,0)])(temp)
#                         flag = True
#                     num = 0
#                     temp = model(temp.to(device),style,'wct')
#                     # while num < 4:
#                     #     temp = model(temp.to(device),style)
#                     #     num += 1
#                     outtemp = temp
#                     if flag:
#                         outtemp = outtemp[:,:,0:min(224, h-x),0:min(224,w-y)]
#                         output[:,:,x:x+min(224,h-x),y:y+min(224,w-y)] = outtemp
#                     else:
#                         output[:,:,x:x+224,y:y+224] = outtemp

#         output_name = output_dir / '{:s}_s_{:s}{:s}'.format(
#             content_path.stem,style_path.stem,args.save_ext)
#         print(output_name)
#         save_image(output, str(output_name))
crop='store_true'
#content_tf = test_transform(0, crop)
content_tf = test_transform(512, crop)
style_tf = test_transform(512, crop)
# for style_path in style_paths:
#     for content_path in content_paths:
    
#         print(content_path)
       
      
#         content_tf1 = content_transform()       
#         content = content_tf(Image.open(content_path).convert("RGB"))

#         h,w,c=np.shape(content)    
#         style_tf1 = style_transform(h,w)
#         style = style_tf(Image.open(style_path).convert("RGB"))

      
#         style = style.to(device).unsqueeze(0)
#         content = content.to(device).unsqueeze(0)
        
#         with torch.no_grad():
#             output= model(content,style,'adain')       
#             #output = model(content)
#         output = torch.cat((content.cpu(), style.cpu(), output.cpu()))
#         #output = inverse_normalize(output,[0.5,0.5,0.5],[0.5,0.5,0.5])
#         output_name = '{:s}/{:s}_stylizing_{:s}{:s}'.format(
#             args.output, splitext(basename(style_path))[0],
#             splitext(basename(content_path))[0], args.save_ext
#         )
 
#         save_image(output, output_name)


# style_path = './data/style/maxresdefault.jpg'
# content_path = './data/content/170508_10_27_41_5DS29248.0.jpg'
      
# content_tf1 = content_transform()       
# content = content_tf(Image.open(content_path).convert("RGB"))

# h,w,c=np.shape(content)    
# style_tf1 = style_transform(h,w)
# style = style_tf(Image.open(style_path).convert("RGB"))


# style = style.to(device).unsqueeze(0)
# content = content.to(device).unsqueeze(0)

# with torch.no_grad():
#     output= model(content,style,'adain')       
#     #output = model(content)
# output = output.cpu()
# style = style.cpu()
# content = content.cpu()
# #output = inverse_normalize(output,[0.5,0.5,0.5],[0.5,0.5,0.5])
# output_name = '{:s}/{:s}_stylizing_{:s}{:s}'.format(
#     'images', splitext(basename(style_path))[0],
#     splitext(basename(content_path))[0], '.png'
# )
# style_name = '{:s}/{:s}{:s}'.format(
#     'images', splitext(basename(style_path))[0], '.png'
# )
# content_name = '{:s}/{:s}{:s}'.format(
#     'images', splitext(basename(content_path))[0], '.png'
# )
# save_image(output, output_name)
# save_image(style, style_name)
# save_image(content, content_name)
# input_pre = './photo/input/in'
# target_pre = './photo/style/tar'
# for i in range(1,61):
#     content_path = input_pre + str(i) + '.png'
#     style_path = target_pre + str(i) + '.png'
#     content_tf1 = content_transform()       
#     content = content_tf(Image.open(content_path).convert("RGB"))

#     h,w,c=np.shape(content)    
#     style_tf1 = style_transform(h,w)
#     style = style_tf(Image.open(style_path).convert("RGB"))

    
#     style = style.to(device).unsqueeze(0)
#     content = content.to(device).unsqueeze(0)
    
#     with torch.no_grad():
#         output= model(content,style,'wct')       
#         #output = model(content)
#     output = output.cpu()
    
#     output = torch.cat((content.cpu(), style.cpu(), output))
#     #output = inverse_normalize(output,[0.5,0.5,0.5],[0.5,0.5,0.5])
#     output_name = '{:s}/output{:s}{:s}'.format(
#         args.output, 
#         str(i), args.save_ext
#     )

#     save_image(output, output_name)
args.content_video = './data/cutBunny.mp4'
import cv2
import imageio
def test_transform(img, size):
    transform_list = []
    # h, w, _ = np.shape(img)
    # if h<w:
    #     newh = size
    #     neww = w/h*size
    # else:
    #     neww = size
    #     newh = h/w*size
    # neww = int(neww//4*4)
    # newh = int(newh//4*4)
    #transform_list.append(transforms.Resize((newh, neww)))
    transform_list.append(transforms.Resize((size,size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
for style_path in style_paths:
    #get video fps & video size
    content_video = cv2.VideoCapture(args.content_video)
    fps = int(content_video.get(cv2.CAP_PROP_FPS))
    content_video_length = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    assert fps != 0, 'Fps is zero, Please enter proper video path'
    pbar = tqdm(total = content_video_length)
    if style_path.suffix in [".jpg", ".png", ".JPG", ".PNG"]:
        # output_video_path = output_dir / '{:s}_stylized_{:s}{:s}'.format(
        #             args.content_video, style_path.stem, args.save_ext)
        output_video_path = 'video_images/data/'+ style_path.stem +'cutBunny.mp4'
        print(output_video_path)
        writer = imageio.get_writer(output_video_path, mode='I', fps=fps)
        #writer = cv2.VideoWriter(str(output_video_path),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (output_width,output_height),1)
        i = 0
        while(True):
            ret, content_img = content_video.read()

            if not ret:
                break

            content = Image.fromarray(content_img)
            img_transform = test_transform(content, args.size)
            content = img_transform(content)
            content = content.to(device).unsqueeze(0)

            style = Image.open(str(style_path)).convert('RGB')
            img_transform = test_transform(style, args.size)
            style = img_transform(style)
            style = style.to(device).unsqueeze(0)

            with torch.no_grad():
                output = model(content,style,'adain')
            output = output.cpu()
            output = output.squeeze(0)
            output_img_path = "video_images/"+style_path.stem+"_{:d}_.jpg".format(i)
            #save_image(output,output_img_path)
            i+=1
            output = output/output.max()
            output = (np.array(output)*255).astype(np.uint8)
            output = np.uint8(output)
            output = np.transpose(output, (1,2,0))
            output = cv2.resize(output, (output_width, output_height))
            #img = imageio.imread(output_img_path)
            writer.append_data(np.array(output))

            pbar.update(1)
        
    content_video.release()
