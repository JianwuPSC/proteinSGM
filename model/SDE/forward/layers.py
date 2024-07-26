# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import string


# From layers.py

def get_act(config):
  """Get activation functions from the config file."""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.ELU()
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.ReLU()
  elif config.model.nonlinearity.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.SiLU()
  else:
    raise NotImplementedError('activation function does not exist!')

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'): 
  """Ported from JAX. """
  # 方差缩放，控制不同层的方差  torch.nn.init.xavier_uniform_() 平替

  def _compute_fans(shape, in_axis=1, out_axis=0): # batch*channel normalize
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init
    
def default_init(scale=1.): # 输出输入维度的init
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000): # position embedding sin/cos 10000
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2 # 整除
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y): #两个数值联合，x,w 
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)]) #a,b,c,d
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)]) #e,f,g,h
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y) #('abcd,dfgh->abcfgh' x,y)


class NIN(nn.Module): # weight, bias init # 可学习的参数
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1) # dim 放最后一维点乘 [B,C,H,W] -> [B,H,W,C]
    y = contract_inner(x, self.W) + self.b # CNN 输出[C,H,W,B]
    return y.permute(0, 3, 1, 2) # dim 放倒数第二维 [B,C,H,W]

    
# From layerspp.py
    
conv1x1 = ddpm_conv1x1
conv3x3 = ddpm_conv3x3
NIN = NIN
default_init = default_init

class AttnBlockpp(nn.Module): # CNN output attention self-head-atten
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x) #[B,C,H,W]
    q = self.NIN_0(h) #[B,C,H,W]
    k = self.NIN_1(h) #[B,C,H,W]
    v = self.NIN_2(h) #[B,C,H,W]

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5)) # CNN transformers attention
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v) # q*k*v
    h = self.NIN_3(h) # attention #[B,C,H,W]
    if not self.skip_rescale:
      return x + h # hidden attention
    else:
      return (x + h) / np.sqrt(2.)


def naive_upsample_2d(x, factor=2): # hight和width 维度扩增两倍
  _N, C, H, W = x.shape
  x = torch.reshape(x, (-1, C, H, 1, W, 1))
  x = x.repeat(1, 1, 1, factor, 1, factor)
  return torch.reshape(x, (-1, C, H * factor, W * factor))

def naive_downsample_2d(x, factor=2): # hight和width 维度缩减两倍
  _N, C, H, W = x.shape
  #print(x.shape)
  x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
  return torch.mean(x, dim=(3, 5))

class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if with_conv:
      self.Conv_0 = conv3x3(in_ch, out_ch)

    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    h = F.interpolate(x, (H * 2, W * 2), 'nearest') #[batch, channel, H * 2, W * 2] F.interpolate 插值扩维
    if self.with_conv:
      h = self.Conv_0(h)

    return h # hidden


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if with_conv:
      self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0) 

    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if self.with_conv:
      x = F.pad(x, (0, 1, 0, 1))
      x = self.Conv_0(x) #[batch, channel, H/2, W/2]
    else:
      x = F.avg_pool2d(x, 2, stride=2) #F.avg_pool2d(x,kernel_size,stride) 平均池化缩维

    return x


class ResnetBlockDDPMpp(nn.Module): # 没有CNN的扩维和缩维
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6) # 将特征通道分成32组
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch) # Dense linear
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape) # Dense weight init
      nn.init.zeros_(self.Dense_0.bias) # Dense bias init
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch) # [32,64]

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x)) # hidden
    h = self.Conv_0(h) # cov3x3
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None] #linear
    h = self.act(self.GroupNorm_1(h)) # swish
    h = self.Dropout_0(h) # Dropout
    h = self.Conv_1(h) # cov3x3
    if x.shape[1] != self.out_ch: # x 维度变为与h same
      if self.conv_shortcut:
        x = self.Conv_2(x) # cov3x3
      else:
        x = self.NIN_0(x) # init
    if not self.skip_rescale:
      return x + h # resnet
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module): # 多了一个上采样和下采样的过程
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6) # groupNorm
    self.up = up
    self.down = down

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch) # 线性层
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape) # init weight
      nn.init.zeros_(self.Dense_0.bias) # init bias

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    if self.up:
      h = naive_upsample_2d(h, factor=2) #上采样
      x = naive_upsample_2d(x, factor=2) #上采样
    elif self.down:
      h = naive_downsample_2d(h, factor=2) #下采样
      x = naive_downsample_2d(x, factor=2) #下采样

    h = self.Conv_0(h) # cov3x3
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None] # linear
    h = self.act(self.GroupNorm_1(h)) # swish
    h = self.Dropout_0(h) # dropout
    h = self.Conv_1(h) # cov3x3

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x) # cov1x1

    if not self.skip_rescale:
      return x + h # h+x resnet
    else:
      return (x + h) / np.sqrt(2.)
