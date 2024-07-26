
import numpy as np
from pathlib import Path
import pickle as pkl
import torch
import argparse
import yaml
from easydict import EasyDict
import time
import shutil
import tqdm
import math
from pyrosetta import *
import rosetta_min.run as rosetta
from torch.utils import tensorboard
from model.utils import save_checkpoint, restore_checkpoint, get_model, recursive_to, Logger
from model.utils import random_mask_batch, get_condition_from_batch, get_mask_all_lengths, get_conditions_from_pdb, get_conditions_random
from model.dataset.dataset import ProteinDataset, PaddingCollate
from model.SDE.forward.ema import ExponentialMovingAverage
import model.SDE.forward.sde_lib as sde_lib
import model.SDE.forward.losses as losses
import model.SDE.sampling.sampling as sampling

#conda activate proteinsgm
#python proteinSGM_inference.py --config=./configs/cond_length.yml --checkpoint=training/cond_length/time/checkpoints/checkpoint_77.pth --pdb ~/data/protein_design/SMURF_protein/example/train_pdb/template_rank.pdb --chain=A --mask_info=9:45 --output_dir=sampling --fastdesign=True --fastrelax=True

def get_args_parser():
    parser = argparse.ArgumentParser('proteinSGM training and inference ', add_help=True)

    parser.add_argument("--config", required=True, type=str, help="dataset config")
    parser.add_argument("--checkpoint", required=True, type=str, help="training checkpoint")
    parser.add_argument("--pdb", default=None, type=str, help="generate seq according pdb ")
    parser.add_argument("--chain", default="A", type=str, help="dataset config")
    parser.add_argument('--mask_info', type=str, default="1:5,10:15")
    parser.add_argument('--select_length', type=bool, default=False)
    parser.add_argument('--length_index', type=int, default=1) # Index starts at 1

    parser.add_argument("--output_dir", required=True, type=str, help="output/model and output")
    parser.add_argument("--tag", default="test", type=str, help="store sample in traning")
    parser.add_argument("--load_model_path", type=str, default=None, help="load model path")

    parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("--n_iter", type=int, default=10, help="number of iter")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")

    parser.add_argument('--index', type=int, default=1) # 1-indexing
    parser.add_argument('--dist_std', type=float, default=2)
    parser.add_argument('--angle_std', type=float, default=20)
    parser.add_argument('--fastdesign', type=bool, default=False)
    parser.add_argument('--fastrelax', type=bool, default=False)

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--device", type=str, default='cuda:0', help="training with cuda:0")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--num_workers", type=int, default=1, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser

def main (args, mode = 'test'):

    if mode == 'test':

        assert not (args.pdb is not None and args.select_length) # pdb/select_length excist only one

        with open(args.config, 'r') as f:
            config = EasyDict(yaml.safe_load(f))

        config.device = args.device
        workdir = Path("sampling", "coords_6d", Path(args.config).stem, Path(args.checkpoint).stem, args.tag)

        print("==============================Initialize model") ########### Initialize model
        score_model = get_model(config)
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        state = restore_checkpoint(args.checkpoint, state, args.device)
        state['ema'].store(state["model"].parameters())
        state['ema'].copy_to(state["model"].parameters())

        print("============================== Model running") ########## Model run

        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales) # 0.01, 100, 2000
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.") # vesde

        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.max_res_num, config.data.max_res_num) #[B,C,N,N]
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

        generated_samples = []

        # Sampling n_iter
        for _ in range(args.n_iter):
            if args.select_length:
                mask = get_mask_all_lengths(config,batch_size=args.batch_size)[args.length_index-1]
                condition = {"length": mask.to(config.device)}
            elif args.pdb is not None:
                condition = get_conditions_from_pdb(args.pdb, config, args.chain, args.mask_info, batch_size=args.batch_size)
                ## out[length]:[B,N,N]; out[ss]:[B,N,N,4:7]; out[inpainting]:[B,N,N],[B,N,N]
            else:
                condition = get_conditions_random(config, batch_size=args.batch_size)
            sample, n = sampling_fn(state["model"], condition) # return (x_mean, sde.N*(n_steps+1))

            generated_samples.append(sample.cpu())

        generated_samples = torch.cat(generated_samples, 0)

        workdir.mkdir(parents=True, exist_ok=True)
        with open(workdir.joinpath("samples.pkl"), "wb") as f:
            pkl.dump(generated_samples, f)

        print("============================== 6d_coords pyrosetta prepare") ########## 6d_coords pyrosetta
        pkl_path = workdir.joinpath("samples.pkl")

        outPath = Path("sampling", "rosetta", args.tag,f"{Path(pkl_path).parent.stem}_index_{args.index}")

        with open(pkl_path, "rb") as f:
            samples = pkl.load(f)

        sample = samples[args.index-1] # 0
        
        # mask sample pose and seq
        msk = np.round(sample[-1]) # 随机取一个sample
        L = math.sqrt(len(msk[msk == 1])) # 没有被mask的氨基酸数目

        if not (L).is_integer():
            raise ValueError("Terminated due to improper masking channel...")
        else:
            L = int(L) # 没有被mask的氨基酸数目
        
        init()

        if args.pdb is not None: # 根据pdb mask 二级结构
            pose = pose_from_pdb(args.pdb)
            seq = pose.sequence()
            res_mask = args.mask_info.split(",")
            for r in res_mask:
                start_idx, end_idx = r.split(":")
                Seq=list(seq)
                Seq[int(start_idx)-1:int(end_idx)-1] = (int(end_idx)-int(start_idx))*"_"
                seq=''.join(Seq)
        else:
            # Initialize sequence of polyalanines and gather constraints
            seq = "A" * L
            pose = None

        # 6d_coords
        npz = {}
        for idx, name in enumerate(["dist", "omega", "theta", "phi"]):
            npz[name] = np.clip(sample[idx][msk == 1].reshape(L, L), -1, 1)

        # Inverse scaling
        npz["dist_abs"] = (npz["dist"] + 1) * 10
        npz["omega_abs"] = npz["omega"] * math.pi
        npz["theta_abs"] = npz["theta"] * math.pi
        npz["phi_abs"] = (npz["phi"] + 1) * math.pi / 2

        print("============================== rosetta relax and fastdesign") ########## rosetta relax and fastdesign
        rosetta.init_pyrosetta()

        for n in range(args.n_iter):
            outPath_run = outPath.joinpath(f"round_{n + 1}")
            if outPath_run.joinpath("final_structure.pdb").is_file():
                continue

            _ = rosetta.run_minimization(
                npz,
                seq,
                pose=pose,
                scriptdir=Path("rosetta_min"),
                outPath=outPath_run,
                angle_std=args.angle_std,  # Angular harmonic std
                dist_std=args.dist_std,  # Distance harmonic std
                use_fastdesign=args.fastdesign,
                use_fastrelax=args.fastrelax,)

        # Create symlink
        if args.fastdesign:
            score_fn = create_score_function("ref2015").score # pyrosetta.create_score_function() 评估能量
            filename = "final_structure.pdb" if args.fastrelax else "structure_after_design.pdb"
        else:
            score_fn = ScoreFunction()
            score_fn.add_weights_from_file(str(Path("rosetta_min").joinpath('data/scorefxn_cart.wts')))
            filename = "structure_before_design.pdb"

        e_min = 9999
        best_run = 0

        for i in range(args.n_iter):
            pose = pose_from_pdb(str(outPath.joinpath(f"round_{i + 1}", filename)))
            e = score_fn(pose)
            if e < e_min:
                best_run = i
                e_min = e

        outPath.joinpath(f"best_run").symlink_to(outPath.joinpath(f"round_{best_run + 1}").resolve(),target_is_directory=True)

        with open(outPath.joinpath("sampling.pkl"), "wb") as f:
            pkl.dump(sample, f)

        ############# finished


if __name__ =='__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'test' # train/evalu

    if mode == 'test':
        main(args, mode=mode)
