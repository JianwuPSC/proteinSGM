
from model.SDE.forward import layers, utils
#model.SDE.forward
import torch.nn as nn
import functools
import torch

ResnetBlockDDPM = layers.ResnetBlockDDPMpp
ResnetBlockBigGAN = layers.ResnetBlockBigGANpp
# Combine = layers.Combine
conv3x3 = layers.conv3x3 # init conv wight+bias
conv1x1 = layers.conv1x1 # init conv wight+bias
get_act = layers.get_act # relu/swish
#get_normalization = normalization.get_normalization # groupNorm
default_initializer = layers.default_init # 方差缩放，控制不同层的方差  torch.nn.init.xavier_uniform_() 平替

class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config) # swish
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config))) # sigmas [0.01,100] 2000个值
    # register_buffer() 用于注册一个缓冲区（buffer），不被学习和优化

    self.nf = nf = config.model.nf # 128
    ch_mult = config.model.ch_mult # [1, 1, 2, 2, 2, 2]
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks # 2
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions # [16]
    dropout = config.model.dropout # 0.1
    resamp_with_conv = config.model.resamp_with_conv # True
    self.num_resolutions = num_resolutions = len(ch_mult) # 6
    self.all_resolutions = all_resolutions = [config.data.max_res_num // (2 ** i) for i in range(num_resolutions)] # [128, 64, 32, 16, 8, 4]

    self.skip_rescale = skip_rescale = config.model.skip_rescale # True
    self.resblock_type = resblock_type = config.model.resblock_type.lower() # 'biggan'
    init_scale = config.model.init_scale #0 

    self.embedding_type = embedding_type = config.model.embedding_type.lower() # positional

    assert embedding_type in ['fourier', 'positional'] # fourier 傅里叶变换 和 位置信息

    modules = [] # 两个线性层 
    embed_dim = nf # 128
    modules.append(nn.Linear(embed_dim, nf * 4)) # 128->512 
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape) # weight init
    nn.init.zeros_(modules[-1].bias) # bias init
    modules.append(nn.Linear(nf * 4, nf * 4)) # 512->512
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    # atten self-atten
    AttnBlock = functools.partial(layers.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)
    # upsample [B,C,H*2,W*2]
    Upsample = functools.partial(layers.Upsample,
                                 with_conv=resamp_with_conv)
    # downsample [B,C,H/2,W/2]
    Downsample = functools.partial(layers.Downsample,
                                   with_conv=resamp_with_conv)
    # resnet DDPM （x+h 过程）无 upsample/downsample
    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)
    # resnet biggan （x+h 过程）有 upsample/downsample
    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)
    
    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block 
    # CNN + ((resnetblock + attenblock)*2 + downsample)*6 + ResnetBlock + AttnBlock + ResnetBlock
    channels = config.data.num_channels # 5
    modules.append(conv3x3(channels, nf)) # CNN 5->128
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions): # range(6)
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks): # range(2)
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    # ((resnetblock + attenblock)*3 + upsample)*6 + GroupNorm + CNN
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    modules = self.all_modules
    # module : 1:linear(128->512), 2:linear(512->512), 3:conv3x3 4:downsample(resnet,atten,downsample/resnet) 
    # 5:(resnet,atten) 6:upsample(resnet,atten,upsample/resnet) 7:groupnorm  8:cov3x3
    m_idx = 0
    # Sinusoidal positional embeddings.
    timesteps = time_cond
    used_sigmas = self.sigmas[time_cond.long()] # sigmas
    temb = layers.get_timestep_embedding(timesteps, self.nf) # time_embedding 作为输入

    temb = modules[m_idx](temb) # module 第一层 linear
    m_idx += 1
    temb = modules[m_idx](self.act(temb)) # module 第二层 activate
    m_idx += 1

    # Downsampling block
    hs = [modules[m_idx](x)] # linear 512+CNN
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb) # resnetblock
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h) # attenblock
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1]) # downsample
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb) # resnetblock
    m_idx += 1
    h = modules[m_idx](h) # attenblock
    m_idx += 1
    h = modules[m_idx](h, temb) # resnetblock
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb) # resnetblock
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h) # AttnBlock
        m_idx += 1

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h) # upsample
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs

    h = self.act(modules[m_idx](h)) # groupNorm
    m_idx += 1
    h = modules[m_idx](h) # CNN
    m_idx += 1

    assert m_idx == len(modules)
    if self.config.model.scale_by_sigma: # 可学习的参数sigma * 参数 scale
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:])))) # timestep reshape
      h = h / used_sigmas

    return h
