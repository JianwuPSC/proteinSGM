from model.SDE.forward import ncsnpp
import torch

# import matplotlib.pyplot as plt
import subprocess
import tempfile

import numpy as np
import torch
import random
from pathlib import Path
from model.dataset.dataset import ProteinDataset, PaddingCollate
from biotite.structure.io import load_structure, save_structure
import biotite.structure as struc
import shutil
import os
import sys

class Logger(object):
    """Writes both to file and terminal"""
    def __init__(self, savepath, mode='a'):
        self.terminal = sys.stdout
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.log = open(os.path.join(savepath, 'logfile.log'), mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

def get_model(config):
    score_model = ncsnpp.NCSNpp(config)
    score_model = score_model.to(config.device)
    score_model = torch.nn.DataParallel(score_model)
    return score_model

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state

def save_checkpoint(ckpt_dir, state):
    saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    else:
        return obj



################################################################################



def random_mask_batch(batch, config): # 蛋白片段 mask
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask_min = config.model.inpainting.mask_min_len # 0.05
    mask_max = config.model.inpainting.mask_max_len # 0.95

    random_mask_prob = config.model.inpainting.random_mask_prob # 0.33
    contiguous_mask_prob = config.model.inpainting.contiguous_mask_prob # 0.33

    lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    prob = random.random()
    if prob < random_mask_prob: # batch随机length mask 0.33 [B,L], 一段蛋白mask
        # Random masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0] # 0.05-0.95 长度中随机取一段 rand, 100
            rand_indices = torch.randperm(l)[:rand] # 0-100

            m = torch.zeros(N)
            m[rand_indices] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    elif prob > 1-contiguous_mask_prob: # 0.67
        # Contiguous masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0] # 100
            index = torch.randint(0, (l - rand).int(), (1,))[0] # 118-100=18中随机取一个数字 14

            m = torch.zeros(N)
            m[index:index + rand] = 1 # 14:114 = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    else:
        mask = torch.ones(B, N) # No masking

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N # 同时为true 或 同时>0 做mask
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch

def selected_mask_batch(batch, mask_info, config): # mask中二级结构区段residule标记为1
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask = torch.zeros(B, N)

    res_mask = mask_info.split(",") # residule
    for r in res_mask:
        if ":" in r:
            start_idx, end_idx = r.split(":")
            mask[:, int(start_idx):int(end_idx)+1] = 1
        else:
            mask[:,int(r)] = 1

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N # 同时为true 或 同时>0 做mask
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch

def get_condition_from_batch(config, batch, mask_info=None):
    batch_size = batch["coords_6d"].shape[0]
    out = {}
    for c in config.model.condition:
        if c == "length": # length mask, < max_res_num true
            lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]
            mask = torch.zeros(batch_size, config.data.max_res_num,
                               config.data.max_res_num).bool()  # B, N, N
            for idx, l in enumerate(lengths):
                mask[idx, :l, :l] = True
            out[c] = mask
        elif c == "ss":
            out[c] = batch["coords_6d"][:,4:7] # dist,w,theta,phi
        elif c == "inpainting":
            if mask_info is not None:
                batch_masked = selected_mask_batch(batch, mask_info, config) # 二级结构区段mask
            else:
                batch_masked = random_mask_batch(batch, config) # 随机区段mask
            out[c] = {
                "coords_6d": batch_masked["coords_6d"],
                "mask_inpaint": batch_masked["mask_inpaint"]
            }

    return recursive_to(out, config.device) # out[length]:[B,N,N]; out[ss]:[B,N,N,4:7]; out[inpainting]:[B,N,N],[B,N,N]

def get_conditions_random(config, batch_size=8): # pdb random select
    # Randomly sample pdbs from dataset
    # Load into dataset/loader and extract info: not very elegant
    paths = list(Path(config.data.dataset_path).iterdir())
    selected = np.random.choice(paths, 100, replace=False) # 每个元素只能被选中一次
    ss_constraints = True if config.data.num_channels == 8 else False
    ds = ProteinDataset(config.data.dataset_path, config.data.min_res_num,
                             config.data.max_res_num, ss_constraints)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                     collate_fn=PaddingCollate(config.data.max_res_num))
    batch = next(iter(dl))
    condition = get_condition_from_batch(config, batch)
    return condition

def get_conditions_from_pdb(pdb, config, chain="A", mask_info=None, batch_size=8): # pdb mask 主要方法
    tempdir = tempfile.TemporaryDirectory()
    # isolate chain
    st = load_structure(pdb)
    st_chain = st[struc.filter_amino_acids(st) & (st.chain_id == chain)]
    save_structure(Path(tempdir.name).joinpath(f"{Path(pdb).stem}_chain_{chain}.pdb"), st_chain) # pdb文件 重新储存到另一个pdb

    ss_constraints = True if config.data.num_channels == 8 else False
    ds = ProteinDataset(tempdir.name, config.data.min_res_num,
                        config.data.max_res_num, ss_constraints) # 读取重新储存的pdb

    dl = torch.utils.data.DataLoader([ds[0]]*batch_size, batch_size=batch_size,
                                     collate_fn=PaddingCollate(config.data.max_res_num))
    batch = next(iter(dl))

    return get_condition_from_batch(config, batch, mask_info=mask_info)

def get_mask_all_lengths(config, batch_size=16): # 部分区段长度mask
    all_lengths = np.arange(config.data.min_res_num, config.data.max_res_num+1) # 40-128

    mask = torch.zeros(len(all_lengths), batch_size, config.data.max_res_num,
                       config.data.max_res_num).bool()  # L, B, N, N

    for idx,l in enumerate(all_lengths):
        mask[idx, :, :l, :l] = True

    return mask

def run_tmalign(path1, path2, binary_path="tm/TMalign", fast=True): # TMalign PDB1.pdb PDB2.pdb -fast
    cmd = [binary_path, path1, path2]
    if fast:
        cmd += ["-fast"]
    result = subprocess.run(cmd, capture_output=True)
    result = result.stdout.decode("UTF-8").split("\n")
    if len(result) < 10: return 0. # when TMalign throws error
    tm = result[13].split(" ")[1].strip()
    return float(tm)

def show_all_channels(sample, path=None, nrows=1, ncols=8): #多张图片整体排列显示
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.1,
                     share_all=True
                     )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    ax_idx = 0
    for s in sample:
        for ch in range(ncols):
            grid[ax_idx].imshow(s[ch])
            ax_idx += 1

    if path:
        plt.savefig(path)

