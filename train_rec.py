# coding=utf-8
from __future__ import absolute_import, division, print_function
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.optim import optimizer
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import save_image
from tqdm import tqdm


from sampler import InfiniteSamplerWrapper
import models.modeling as modeling
from functions import calc_content_loss, calc_style_loss, train_transform, normal, inverse_normalize
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--log_name",  default=localtime,
                    help="Log dir of this run. Used for monitoring.")
parser.add_argument("--pretrained_dir", type=str, default="./models/vit.pth",
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--save_every", default=5000, type=int,
                    help="Save checkpoints every so many steps."
                    "Will always save one checkpoint at the end of training.")                

parser.add_argument("--img_size", default=256, type=int,
                    help="Resolution size")
parser.add_argument("--train_batch_size", default=14, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_every", default=100, type=int,
                    help="Save training samples every so many steps."
                    "Will always run one evaluation at the end of training.")

parser.add_argument("--lr", default=5e-4, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--lr_decay", default=1e-5, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=480000, type=int,
                    help="Total number of training iterations to perform.")
parser.add_argument("--warmup_steps", default=1e4, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--accum_steps", default=4, type=int,
                    help="Step of training to perform gradient accumulation.")
parser.add_argument("--debug", action='store_true', default=False,
                    help="start with debug mode, without any output, log and checkpoints.")
args = parser.parse_args()

# Setup CUDA, GPU & distributed training

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def warmup_learning_rate(optimizer, iteration_count):#0.0002
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

if args.debug == False:
    writer = SummaryWriter(log_dir='./logs/'+args.log_name)
    if not os.path.exists('./ckpts/'+ args.log_name):
        os.mkdir('./ckpts/'+ args.log_name)
    if not os.path.exists('./logs/'+args.log_name):
        os.mkdir('./logs/'+args.log_name)
    if not os.path.exists('./training_images/'+args.log_name):
        os.mkdir('./training_images/'+args.log_name)

train_dataset = FlatFolderDataset("/home/weihuapeng/dataset/coco2014", train_transform(args.img_size))
train_loader = data.DataLoader(
    train_dataset, batch_size=args.train_batch_size,sampler=InfiniteSamplerWrapper(train_dataset),
    num_workers=args.train_batch_size)

style_dataset = FlatFolderDataset("/home/weihuapeng/dataset/wikiart", train_transform(args.img_size))
style_loader = data.DataLoader(
    style_dataset,batch_size = args.train_batch_size, sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.train_batch_size)

#my_Vit  ViT-B_16_h ViT-B_16
#config = modeling.CONFIGS['my_Vit']
config = modeling.CONFIGS['ViT-B_16']
model = modeling.VisionTransformer(config)

print("Model created!")
#model.transformer.embeddings.patch_embeddings = Conv2d(3,768,8,8)
# model.transformer.load_state_dict(torch.load('./ckpts/exp9_random_ViT_wct/vit_iter_160000.pth'))
# model.transformerdecoder.load_state_dict(torch.load('./ckpts/exp9_random_ViT_wct/decoder_iter_160000.pth'))
model.to(args.device)
model.transformer.load_state_dict(torch.load(args.pretrained_dir))
model.transformerdecoder.load_state_dict(torch.load('./ckpts/exp16_pre_train_ViT_wct_mse/decoder_iter_15000.pth'))
for inx, para in enumerate(model.transformer.parameters()):
    para.requires_grad = False

args.DP = True
if args.DP:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    optimizer = torch.optim.Adam([ 
                              #{'params': model.module.transformer.parameters()},
                              {'params': model.module.transformerdecoder.parameters()},    
                              #{'params': model.module.transformer.embeddings.patch_embeddings.parameters()},    
                            ], 
                              lr = args.lr)
else:
    optimizer = torch.optim.Adam([ 
                              #{'params': model.transformer.parameters()},
                              {'params': model.transformerdecoder.parameters()},    
                              #{'params': model.module.transformer.embeddings.patch_embeddings.parameters()},    
                            ], 
                              lr = args.lr)
mseloss = nn.MSELoss()
smoothl1loss = nn.SmoothL1Loss()




vgg = modeling.vgg
vgg.load_state_dict(torch.load('./models/vgg_normalised.pth'))
vgg = nn.Sequential(*list(vgg.children())[:44])
with torch.no_grad():
    loss_net = modeling.Net(vgg)
loss_net.to(args.device)
train_iter = iter(train_loader)
style_iter = iter(style_loader)


for i in tqdm(range(15000, args.num_steps)):
    if i < args.warmup_steps:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    try:
        content_image = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        content_image = next(train_iter)

    try:
        style_image = next(style_iter)
    except StopIteration:
        style_iter = iter(style_loader)
        style_image = next(style_iter)

    content_image = content_image.to(device)
    style_image = style_image.to(device)

    # Ics = model(content_image,style_image,'adain')
    Icc = model(content_image)
    # #Icc = model(content_image)#rec
    Iss = model(style_image)

    #style_feats, content_feats, Ics_feats, Icc_feats, Iss_feats= loss_net(style_image, content_image, Ics, Icc, Ics)

    #loss_c = calc_content_loss(normal(content_feats[-1]), normal(Ics_feats[-1])) + calc_content_loss(normal(content_feats[-2]), normal(Ics_feats[-2]))
    # loss_c = calc_content_loss(content_feats[-1], Ics_feats[-1]) + calc_content_loss(content_feats[-2], Ics_feats[-2])
    # #loss_c = calc_content_loss(content_feats[-1], Icc_feats[-1])
    # loss_s = calc_style_loss(Ics_feats[0], style_feats[0])
    # for j in range(1, 5):
    #     loss_s += calc_style_loss(Ics_feats[j], style_feats[j])
    # loss_lambda1 = calc_content_loss(Icc,content_image)+calc_content_loss(Iss,style_image)
    # loss_lambda2 = calc_content_loss(Icc_feats[0], content_feats[0])+calc_content_loss(Iss_feats[0], style_feats[0])
    # for j in range(1, 5):
    #     loss_lambda2 += calc_content_loss(Icc_feats[j], content_feats[j])+calc_content_loss(Iss_feats[j], style_feats[j])
    loss_rec = mseloss(Icc, content_image) + mseloss(Iss, style_image)
    
    # loss_rec = smoothl1loss(Icc, content_image) + smoothl1loss(Iss, style_image)
    #loss = 10 * loss_s + 10 * loss_c# + (loss_lambda1 * 70) + (loss_lambda2 * 1) 
    loss = loss_rec
    # print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
    #     #      ,"-l1:",loss_lambda1.sum().cpu().detach().numpy(),"-l2:",loss_lambda2.sum().cpu().detach().numpy()
    #     )
    print(loss.sum().cpu().detach().numpy(),"-rec:",loss_rec.sum().cpu().detach().numpy()
        #      ,"-l1:",loss_lambda1.sum().cpu().detach().numpy(),"-l2:",loss_lambda2.sum().cpu().detach().numpy()
        )
    #loss = loss / args.accum_steps
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    
    
    # if (i+1) % args.accum_steps == 0 or (i+1) == args.num_steps:
    #     optimizer.step() 
    #     optimizer.zero_grad()
        

    if args.debug == False:
        # writer.add_scalar('loss_c',loss_c.item(),i+1)
        writer.add_scalar('loss_rec',loss_rec.item(),i+1)
        # writer.add_scalar('loss_s',loss_s.item(),i+1)
        # writer.add_scalar('loss_lambda1',loss_lambda1.item(),i+1)
        # writer.add_scalar('loss_lambda2',loss_lambda2.item(),i+1)
        writer.add_scalar('loss',loss.item(),i+1)
 
    if ((i+1) % args.eval_every == 0 or (i+1)== args.num_steps) and (args.debug == False):
        # saving eval images
        # content_image = inverse_normalize(content_image,[0.5,0.5,0.5],[0.5,0.5,0.5])
        # style_image = inverse_normalize(style_image,[0.5,0.5,0.5],[0.5,0.5,0.5])
        # Ics = inverse_normalize(Ics,[0.5,0.5,0.5],[0.5,0.5,0.5])
        output_image = torch.cat((content_image, style_image, Icc, Iss))
        #output_image = torch.cat((content_image,Icc))
        output_image = output_image.cpu()
        save_path = str('./training_images/'+args.log_name + "/iter_"  + str(i+1) + "_th image.jpg")
        save_image(output_image, save_path, nrow=args.train_batch_size)


    if ((i+1) % args.save_every == 0 or (i+1)== args.num_steps) and (args.debug == False):
        if args.DP:
            # saving checkpoints
            print("---------saving ckpts----------")
            state_dict = model.module.transformerdecoder.state_dict()
            #state_dict = model.transformerdecoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                        '{:s}/decoder_iter_{:d}.pth'.format('./ckpts/'+ args.log_name,
                                                                i + 1))
            # state_dict = model.module.transformer.state_dict()
            # #state_dict = model.transformer.state_dict()
            # for key in state_dict.keys():
            #     state_dict[key] = state_dict[key].to(torch.device('cpu'))
            # torch.save(state_dict,
            #             '{:s}/vit_iter_{:d}.pth'.format('./ckpts/'+ args.log_name,
            #                                                     i + 1))
        else:
            print("---------saving ckpts----------")
            #state_dict = model.module.transformerdecoder.state_dict()
            state_dict = model.transformerdecoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,
                        '{:s}/decoder_iter_{:d}.pth'.format('./ckpts/'+ args.log_name,
                                                                i + 1))
            #state_dict = model.module.transformer.state_dict()
            # state_dict = model.transformer.state_dict()
            # for key in state_dict.keys():
            #     state_dict[key] = state_dict[key].to(torch.device('cpu'))
            # torch.save(state_dict,
            #             '{:s}/vit_iter_{:d}.pth'.format('./ckpts/'+ args.log_name,
            #                                                     i + 1))            
if args.debug == False:
    writer.close()
