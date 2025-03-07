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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from model.SDE.forward import utils as mutils
import biotite.structure as struc
import random

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr, # 0.0001
                  warmup=config.optim.warmup, # 5000
                  grad_clip=config.optim.grad_clip): # 1
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn

# Randomly dropout block adjacencies
def block_dropout(coords_6d, ss_indices, block_dropout=0.2): # a-helix,b-fold random mask
  for idx in range(len(ss_indices)):
    if ss_indices[idx] == '': continue # no secondary structure annotation found
    ss_idx = ss_indices[idx].split(",")
    indices_for_dropout = [b for b in ss_idx if random.random() < block_dropout]
    for i in indices_for_dropout:
      start, end = [int(x) for x in i.split(":")]
      coords_6d[idx,4:7, :, start:end] = 0 # [batch,8,118,118]
      coords_6d[idx,4:7, start:end, :] = 0 # [batch,8,118,118]

  return coords_6d # [batch,8,118,118]

def get_sde_loss_fn(sde, train, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """

  def loss_fn(model, batch, condition=None):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """

    coords_6d = batch["coords_6d"] # coords_6d: [batch,8,118,118]
    mask_pair = batch["mask_pair"] # mask_pair: [batch,118,118] True/False

    if "ss" in condition:
      coords_6d = block_dropout(coords_6d, batch["ss_indices"]) # Dropout block adjacencies # ss_indices:'6:15,21:34,49:61,98:115'
      
    # 1. 随机生成batch_size个，random float t
    score_fn = mutils.get_score_fn(sde, model, train=train) # model VPSDE/VESDE
    t = torch.rand(coords_6d.shape[0], device=coords_6d.device) * (sde.T - eps) + eps # 产生随即张量 sde.T=1
    # 2. 根据重参数技巧采样出分布 p_t(x) 的一个随机样本perturbed_data
    z = torch.randn_like(coords_6d) # Z(0,1)
    mean, std = sde.marginal_prob(coords_6d, t) # mean与coords相关，std与t相关
    perturbed_data = mean + std[:, None, None, None] * z # pertubed_data 
    # mask information
    conditional_mask = torch.ones_like(coords_6d).bool() # mask 长度，二级结构，修补位点
    if condition is not None:
      for c in condition:
        if c == "length":
          conditional_mask[:,-1] = False
        elif c == "ss":
          conditional_mask[:,4:7] = False
        elif c == "inpainting":
          mask_inpaint = batch["mask_inpaint"].unsqueeze(1)
          conditional_mask = conditional_mask * mask_inpaint

    mask = mask_pair.unsqueeze(1) * conditional_mask # total mask [batch,1,118,118] * [batch,8,118,118]
    num_elem = mask.reshape(mask.shape[0], -1).sum(dim=-1)

    perturbed_data = torch.where(mask, perturbed_data, coords_6d) # 根据mask(bool)信息筛选perturbed_data, false位点输出coords_6d
    # 3. 将当前的加噪样本和时间输入score network 中预测分数
    score = score_fn(perturbed_data, t)
    # 4. 计算 score matching loss
    losses = torch.square(score * std[:, None, None, None] + z) * mask
    losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
    losses = losses / (num_elem + 1e-8) # 1e-8 added to prevent nan when num_elem TODO: Fix masking to prevent this
    loss = torch.mean(losses)

    return loss

  return loss_fn

def get_step_fn(sde, train, optimize_fn=None):
  """Create a one-step training/evaluation function.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.
  Returns:
    A one-step function for training or evaluation.
  """
  loss_fn = get_sde_loss_fn(sde, train)

  def step_fn(state, batch, condition=None):
    """Running one step of training or evaluation.
    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.
    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.
    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch, condition)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch, condition)
        ema.restore(model.parameters())

    return loss

  return step_fn
