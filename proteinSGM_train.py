
from pathlib import Path
import pickle as pkl
import torch
import argparse
import yaml
from easydict import EasyDict
import time
import shutil
from torch.utils import tensorboard
from model.utils import save_checkpoint, restore_checkpoint, get_model, recursive_to, Logger
from model.utils import random_mask_batch, get_condition_from_batch
from model.dataset.dataset import ProteinDataset, PaddingCollate
from model.SDE.forward.ema import ExponentialMovingAverage
import model.SDE.forward.sde_lib as sde_lib
import model.SDE.forward.losses as losses
import model.SDE.sampling.sampling as sampling

#conda activate proteinsgm
#python proteinSGM_train.py --dataset_path='/home/wuj/data/tools/proteinSGM/data/cath/dompdb' --config=configs/cond_length.yml --batch_size=8 --sampler_number=2000000 --epochs=2000000 --output_dir='training' --sample_store=False

def get_args_parser():
    parser = argparse.ArgumentParser('proteinSGM training and inference ', add_help=True)

    parser.add_argument("--dataset_path", required=True, type=str, help="pathway in train dataset")
    #parser.add_argument("--sample_names", required=True, nargs='+', action='store', type=str, help="train sample namses")
    parser.add_argument("--config", required=True, type=str, help="dataset config")

    parser.add_argument("--output_dir", required=True, type=str, help="output/model and output")
    parser.add_argument("--sample_store", default=False, type=bool, help="store sample in traning")
    parser.add_argument("--load_model_path", type=str, default=None, help="load model path")

    parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
    parser.add_argument("--sampler_number", type=int, default=100000, help="number of sampler numer")
    parser.add_argument("--epochs", type=int, default=100000, help="number of epochs")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--device", type=str, default='cuda:0', help="training with cuda:0")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--save_checkpoint", type=bool, default=True, help="Save checkpoint: true or false")
    parser.add_argument("--save_prediction", type=bool, default=True, help="Save prediction: true or false")
    parser.add_argument("--n_ensembles", type=int, default=3, help="Number of models in ensemble")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--num_workers", type=int, default=1, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    return parser

def main (args, mode = 'train'):

    if mode == 'train':

        print("===============================Dataloading") ############### Dataloading
        with open(args.config, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        print(f"Train config {args.config} loading")

        ss_constraints = True if config.data.num_channels == 8 else False
        # Dataset
        dataset = ProteinDataset(args.dataset_path, config.data.min_res_num,
                             config.data.max_res_num, ss_constraints)
        # coords:[118,118,3]  coords_6d:[8,118,118]  aa:[118]  mask_pair:[118,118] true/false  ss_indices:'6:15,21:34,49:61,98:115'
        print(f"Train Dataset {args.dataset_path} loading")

        train_size = int(0.95 * len(dataset))
        eval_size = len(dataset) - train_size
        # DataLoader
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size],
                                                      generator=torch.Generator().manual_seed(config.seed))

        train_sampler = torch.utils.data.RandomSampler(train_dataset,replacement=True,
                                                       num_samples=args.sampler_number*config.training.batch_size)
        train_dataload = torch.utils.data.DataLoader(train_dataset,sampler=train_sampler,batch_size=config.training.batch_size,
                                              collate_fn=PaddingCollate(config.data.max_res_num))
        train_iter = iter(train_dataload)

        eval_sampler = torch.utils.data.RandomSampler(eval_dataset,replacement=True,
                                                      num_samples=args.sampler_number*config.training.batch_size)
        eval_dataload = torch.utils.data.DataLoader(eval_dataset,sampler=eval_sampler,batch_size=config.training.batch_size,
                                              collate_fn=PaddingCollate(config.data.max_res_num))
        eval_iter = iter(eval_dataload)
        print(f"{args.sampler_number*config.training.batch_size} samples in traing/test iter and batch_size is {config.training.batch_size}")

        print("==============================Initialize model") ########### Initialize model
        # score model
        score_model = get_model(config)
        # ema
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        print("============================== Constract workdir and logdir") ########## Constract workdir and logdir
        workdir = Path(args.output_dir, Path(args.config).stem, 'time')
        #workdir = Path(args.output_dir, Path(args.config).stem, time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        workdir.mkdir(exist_ok=True,parents=True)
        shutil.copy(args.config, workdir.joinpath("config.yml"))
        sample_dir = workdir.joinpath("samples")
        sample_dir.mkdir(exist_ok=True)

        tb_dir = workdir.joinpath("tensorboard")
        tb_dir.mkdir(exist_ok=True)
        writer = tensorboard.SummaryWriter(tb_dir)
        logger = Logger(tb_dir)

        checkpoint_dir = workdir.joinpath("checkpoints")
        checkpoint_meta_dir = workdir.joinpath("checkpoints-meta", "checkpoint.pth") # resume
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_meta_dir.parent.mkdir(exist_ok=True)

        if checkpoint_meta_dir.is_file():
            state = restore_checkpoint(checkpoint_meta_dir, state, config.device) # utils.restore_checkpoint
            initial_step = int(state['step'])
        else:
            initial_step = 0

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

        optimize_fn = losses.optimization_manager(config) # training forward
        train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn)
        eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn)

        if args.sample_store: # sampling
            sampling_shape = (config.training.batch_size, config.data.num_channels,
                              config.data.max_res_num, config.data.max_res_num) #[B,N,L,L]
            sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)

        # train one trainbatch
        for step in range(initial_step, args.epochs + 1):
            train_batch = recursive_to(next(train_iter), config.device) # 递归，将数据集按所需类型整合
            train_batch = random_mask_batch(train_batch, config) # protein mask
            train_loss = train_step_fn(state, train_batch, condition=config.model.condition)

            if step % config.training.log_freq == 0:
                writer.add_scalar("training_loss", train_loss, step)

            # Save a temporary checkpoint to resume training after pre-emption periodically
            if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
                save_checkpoint(checkpoint_meta_dir, state)

            # Report the loss on an evaluation dataset periodically
            if step % config.training.eval_freq == 0: # 整数倍 % 除法取余数
                eval_batch = recursive_to(next(eval_iter), config.device)
                eval_batch = random_mask_batch(eval_batch, config)
                eval_loss = eval_step_fn(state, eval_batch, condition=config.model.condition)
                writer.add_scalar("eval_loss", eval_loss.item(), step)
            
            # Save a checkpoint periodically and generate samples if needed
            if step != 0 and step % config.training.snapshot_freq == 0 or step == args.epochs:
                # Save the checkpoint.
                save_step = step // config.training.snapshot_freq # // 除法取整数
                save_checkpoint(checkpoint_dir.joinpath(f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                #if config.training.snapshot_sampling: # true
                if args.sample_store:
                    ema.store(score_model.parameters()) # 上边都是正向loss
                    ema.copy_to(score_model.parameters()) # 上边都是正向loss
                    condition = get_condition_from_batch(config, eval_batch)
                    sample, n = sampling_fn(score_model, condition=condition)
                    ema.restore(score_model.parameters())
                    this_sample_dir = sample_dir.joinpath(f"iter_{step}")
                    this_sample_dir.mkdir(exist_ok=True)

                    with open(str(this_sample_dir.joinpath("sample.pkl")), "wb") as fout:
                        pkl.dump(sample.cpu(), fout)

            if step % config.training.eval_freq == 0:
                logger.write('step:{},train_loss:{},eval_loss:{}\n'.format(step,train_loss.item(),eval_loss.item()))


if __name__ =='__main__':

    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'train' # train/evalu

    if mode == 'train':
     #   print('========  Train  =============================================================')
        main(args, mode=mode)
    #else:
    #    main(args, mode=mode)
