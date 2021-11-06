# coding=utf-8
# modified from https://github.com/jeonsworld/ViT-pytorch
from __future__ import absolute_import, division, print_function

import copy
import logging
import math
from os.path import join as pjoin
from typing import DefaultDict

import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, Conv2d, CrossEntropyLoss, Dropout,
                      GroupNorm, LayerNorm, Linear, Softmax)
from torch.nn.modules.utils import _pair

import models.configs as configs
#from models.modeling_resnet import ResNetV2
import models.sinusoidal_positional_embedding as sinusoidal_positional_embedding
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"

M_ATTENTION_Q = "MultiHeadDotProductAttention_2/query"
M_ATTENTION_K = "MultiHeadDotProductAttention_2/key"
M_ATTENTION_V = "MultiHeadDotProductAttention_2/value"
M_ATTENTION_OUT = "MultiHeadDotProductAttention_2/out"

FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
M_ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        mixed_query_layer = self.query(q)#[B, p^2, hidden_size] -> [B, p^2, all_head_size]
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)

        query_layer = self.transpose_for_scores(mixed_query_layer) #[B, p^2, hiden_size] -> [B, p^2, head_num, attn_head_size] -> [B, head_num, p^2, attn_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #[B, head_num, p^2, p^2]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) #[B, head_num, p^2, p^2] -> [B, head_num, p^2, attn_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()#[B, p^2, head_num, attn_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)#[B, p^2, hidden_size]
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, pe_mode = None):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)


        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        if pe_mode == 'learnable' or pe_mode == None:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        elif pe_mode == 'sin':
            self.position_embeddings = sinusoidal_positional_embedding.SinusoidalPositionalEncoding(config.hidden_size)
        self.pe_mode = pe_mode
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x) #[B, hidden_size, patchsize, pathcsize]
        x = x.flatten(2) #[B, hidden_size, patchsize*patchsize]
        x = x.transpose(-1, -2) #[B, patchsize*patchsize, hidden_size]
        #x = torch.cat((cls_tokens, x), dim=1) #[B, patchsize*patchsize+1, hidden_size]
        
        if self.pe_mode == 'learnable':
            # 对PE进行插值
            pe = F.interpolate(self.position_embeddings.transpose(-1,-2), mode='nearest',size=x.shape[1]).transpose(-1,-2)
            embeddings = x + pe
        elif self.pe_mode == 'sin':
            embeddings = self.position_embeddings(x)
        else:
            embeddings = x
        embeddings = self.dropout(embeddings)
        return embeddings
class Tail(nn.Module):
    def __init__(self, config, img_size, out_channel=3, upsample=None):
        super().__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

                #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.upsample = upsample
        self.config = config
        if upsample=='mlp':
            self.tail = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
        else:
            if self.config.patches.size == (8, 8):
                self.tail = nn.Sequential(
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(config.hidden_size, 256, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    #nn.Upsample(scale_factor=2, mode='nearest'),##up
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 128, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 128, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 64, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 64, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 3, (3, 3)),
                )
            elif self.config.patches.size == (16,16):
                self.tail = nn.Sequential(
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(config.hidden_size, 256, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),##up
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 256, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(256, 128, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 128, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(128, 64, (3, 3)),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 64, (3, 3)),
                    nn.ReLU(),
                    nn.ReflectionPad2d((1, 1, 1, 1)),
                    nn.Conv2d(64, 3, (3, 3)),
                )

    def forward(self, x):

        # x: NxHWxC
        if self.upsample == 'mlp':
            x = self.tail(x)
            B, P, H = x.size()
            p = int(math.sqrt(P))
            patch_size = int(math.sqrt(H/3))
            result = x.view(-1, p, p, patch_size, patch_size, 3).permute(0,5,1,3,2,4).contiguous().view(-1,3,patch_size*p,patch_size*p)
        else:
            x = x.transpose(-1, -2)
            B, H, P = x.size()
            p = int(math.sqrt(P))
            x = x.view(B, H, p, p)
            result = self.tail(x)#B, 768, 14, 14
        return result


class EncoderBlock(nn.Module):
    def __init__(self, config, vis):
        super(EncoderBlock, self).__init__()
        self.hidden_size = config.hidden_size

        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x, x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class DecoderBlock(nn.Module):
    def __init__(self, config, vis):
        super(DecoderBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.multihead_attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.multihead_attn = Attention(config, vis)

    def forward(self, x, memory):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, x, x)
        x = x + h

        h = x
        q = self.multihead_attention_norm(x)
        k = self.multihead_attention_norm(memory)
        v = self.multihead_attention_norm(memory)
        # k = self.multihead_attention_norm(x)
        # v = self.multihead_attention_norm(x)

        x, weights = self.multihead_attn(q, k, v)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = EncoderBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    def nst(self,hidden_states):
        attn_weights = []
        encoded = []
        b,x,c = hidden_states.size()
        xx = int(math.sqrt(x))
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
            a = hidden_states.clone()
            encoded.append(a.view(-1,xx,xx,768).permute(0,3,1,2))
        
        a = self.encoder_norm(hidden_states)
        encoded.append(a.view(-1,xx,xx,768).permute(0,3,1,2))
        return encoded, attn_weights

class Decoder(nn.Module):
    def __init__(self, config, vis):
        super(Decoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.decoder_norm = LayerNorm(config.hidden_size, eps=1e-6)   
        #self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        for _ in range(config.transformer["num_layers"]):
            layer = DecoderBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, memory):
        attn_weights = []
        #hidden_states = hidden_states + self.position_embeddings
        for layer_block in self.layer:

            hidden_states, weights = layer_block(hidden_states, memory)
            if self.vis:
                attn_weights.append(weights)
        decoded = self.decoder_norm(hidden_states)

        return decoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, pe_mode = None):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size, pe_mode=pe_mode)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    def nst(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder.nst(embedding_output)
        return encoded, attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, config, img_size, vis, upsample=None):
        super(TransformerDecoder, self).__init__()
        self.decoder = Decoder(config, vis)
        self.tail = Tail(config, img_size=img_size, upsample=upsample)
    def forward(self, encoded, encoded1=None):
        if encoded1 is not None:
            decoded, attn_weights = self.decoder(encoded,encoded1)
        else:
            decoded, attn_weights = self.decoder(encoded,encoded)


        #decoded = whiten_pre(decoded)


        output = self.tail(decoded)
        # output = self.tail(dncoded)
        return output,[]#, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, pe_mode = None, upsample=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis, pe_mode=pe_mode)
        self.head = Linear(config.hidden_size, num_classes)
        self.transformerdecoder = TransformerDecoder(config, img_size, vis, upsample=upsample)

    def forward(self, x, y=None, mode=None):

        if y is not None:
            fc, _ = self.transformer(x)
            fs, _ = self.transformer(y)

            if mode=='wct':
                a = 1
                fcs, fs = wct(fc,fs)
                fcs = a*fcs + (1-a)*fc
            elif mode=='wct_within_heads':
                fcs, fs = wct_within_heads(fc,fs)
                
            elif mode=='wct_transpose':
                fcs, fs = wct_transpose(fc,fs)

            elif mode=='adain':
                fcs, fs = adain(fc,fs)
            elif mode=='whiten':
                fcs = whiten_pre(fc)
            elif mode=='attention':
                out, _ = self.transformerdecoder(fc,fs)
                return out

            elif mode==None:
                fcs = fc
            
            out, _ = self.transformerdecoder(fcs)
            return out
        else:

            encoded, attn_weights = self.transformer(x)
            decoded, attn_weights= self.transformerdecoder(encoded)
        return decoded

def whiten_pre(fc):

    b, x, c = fc.size()
    fcs = torch.rand_like(fc)
    for i in range(b):
        fc_ = fc[i].view(x,c).transpose(-1,-2)

        fcs_ = whiten(fc_)
        fcs[i] = fcs_.transpose(-1,-2).view(x,c).to(fc.device)
    
    return fcs
def whiten(cF):
    s_dtype = cF.dtype
    device = cF.device
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1)# + torch.eye(cFSize[0]).double().to(device)
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)
    #c_d = (c_e[0:k_c])
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))

    cF = cF.to(step2.dtype)
    #whiten_cF = torch.mm(c_v[:,0:k_c],cF)
    whiten_cF = torch.mm(step2,cF)
    return whiten_cF.to(s_dtype)        
def whiten_and_color(cF,sF): #(c,hw)
    s_dtype = cF.dtype
    device = cF.device
    cFSize = cF.size() 
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean # centering

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1)# + torch.eye(cFSize[0]).double().to(device) #(cF * cF.T) / (hw-1) + I_c

    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1) #+ torch.eye(sFSize[0]).double().to(device)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    #c_d = (c_e[0:k_c])
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))

    cF = cF.to(step2.dtype)
    #whiten_cF = torch.mm(c_v[:,0:k_c],cF)
    whiten_cF = torch.mm(step2,cF)


    s_d = (s_e[0:k_s]).pow(0.5)
    t_dtype = whiten_cF.dtype
    s_v = s_v.to(t_dtype)
    s_d = s_d.to(t_dtype)

    targetFeature = torch.mm(
        torch.mm(
            torch.mm(s_v[:,0:k_s], 
            torch.diag(s_d)),
        (s_v[:,0:k_s].t())),
    whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature = targetFeature.to(s_dtype)
    return targetFeature.to(s_dtype)
    #return whiten_cF.to(s_dtype)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std

def feature_transform(feature):

    feature = feature.transpose(-1,-2)
    return feature
def feature_reverse(feature):

    feature = feature.transpose(-1,-2)
    return feature

def adaptive_instance_normalization(content_feat, style_feat):

    assert (content_feat.size()[:2] == style_feat.size()[:2])

    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def adain(fc,fs):

    fc_ = feature_transform(fc)
    fs_ = feature_transform(fs)
    fcs = adaptive_instance_normalization(fc_,fs_)
    fcs = feature_reverse(fcs)
    #fs_ = feature_reverse(fs_)

    return fcs,fs

def wct_within_heads(fc,fs):
    b, x, c = fc.size()
    fc_ = fc.contiguous().view(x,12,c//12).permute(1,2,0).cpu() # (12,c//12,x)
    fs_ = fs.contiguous().view(x,12,c//12).permute(1,2,0).cpu()
    fcs = torch.rand_like(fc_)
    for i in range(fc_.size(0)):
        #print(fc_[i].size())
        fcs[i] = whiten_and_color(fc_[i],fs_[i])
    fcs = fcs.permute(2,0,1).view(b,x,c).to(fc.device)
    return fcs,fs

def wct(fc,fs):

    b, x, c = fc.size()
    fcs = torch.rand_like(fc)
    bx, xs,cs = fs.size()
    for i in range(b):
        fc_ = fc[i].view(x,c).transpose(-1,-2)
        fs_ = fs[i].view(xs,cs).transpose(-1,-2)

        fcs_ = whiten_and_color(fc_,fs_)
        fcs[i] = fcs_.transpose(-1,-2).view(x,c).to(fc.device)
    
    return fcs,fs
def wct_transpose(fc,fs):

    b, x, c = fc.size()
    fcs = torch.rand_like(fc)

    for i in range(b):
        fc_ = fc[i].view(x,c).cpu() 
        fs_ = fs[i].view(x,c).cpu()

        fcs_ = whiten_and_color(fc_,fs_)
        fcs[i] = fcs_.view(x,c).to(fc.device)
    return fcs,fs
class VisionTransformer_eval(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer_eval, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        self.transformerdecoder = TransformerDecoder(config, img_size, vis)

    def forward(self, content, style, mode=None):
        #[B,  p^2 , c]

        fc, _ = self.transformer(content)
        fs, _ = self.transformer(style)

        if mode=='wct':
            fcs, fs = wct(fc,fs)

        elif mode=='wct_within_heads':
            fcs, fs = wct_within_heads(fc,fs)
        
        elif mode=='adain':
            fcs, fs = adain(fc,fs)

        #elif mode=='attention':
            

        elif mode==None:
            fcs = fc

        #out, _ = self.transformerdecoder(fcs)
        out, _ = self.transformerdecoder.forward_eval(fcs,fs)

        return out

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    def forward(self, style, content, Ics, Icc, Iss):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        Ics_feats = self.encode_with_intermediate(Ics)
        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)

        return style_feats, content_feats, Ics_feats, Icc_feats, Iss_feats


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
    'my_Vit' : configs.get_my_config(),
    'ViT-B_16_h' : configs.get_b16_h_config()
}

if __name__ == '__main__':
    config = CONFIGS['my_Vit']
    model = VisionTransformer(config)
    a = torch.randn(1,3,512,512)
    b = model(a)
    tail = Tail(config,512,3)

