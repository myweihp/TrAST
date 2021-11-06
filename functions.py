import torch
import torch.nn as nn
from torchvision import transforms


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal(feat, eps=1e-5):
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 
mse_loss = nn.MSELoss()
def calc_content_loss(input, target):
      assert (input.size() == target.size())
      #assert (target.requires_grad is False)
      return mse_loss(input, target)

def calc_style_loss( input, target):
    assert (input.size() == target.size())
    #assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), GramMatrix()(target))
        return(out)


def train_transform(size=224):
    transform_list = [
        transforms.Resize((512,512)),
        transforms.RandomCrop(size=(size, size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(transform_list)

# def train_transform():
#     transform_list = [
#         transforms.ToTensor()
#     ]
#     return transforms.Compose(transform_list)
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t = t.mul(s).add(m)
    return tensor