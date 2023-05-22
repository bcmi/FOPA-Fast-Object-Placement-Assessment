import os
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import torch.backends.cudnn as torchcudnn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchvision import transforms
    
import argparse
import random
import network
import tensorboard_logger as tb_logger
import torch.nn as nn

from backbone.ResNet import pretrained_resnet18_4ch
from config import arg_config, proj_root
from data.OBdataset import create_loader
from utils.misc import AvgMeter, construct_path_dict, make_log, pre_mkdir

parser = argparse.ArgumentParser()
parser.add_argument('--ex_name', type=str, default=arg_config["ex_name"])
parser.add_argument('--alpha', type=float, default=16.)
parser.add_argument('--resume', type=bool, help='resume from checkpoint')

user_args = parser.parse_args()
datetime_str = str(datetime.now())
datetime_str = '-'.join(datetime_str.split())
user_args.ex_name += datetime_str

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# set random seed
setup_seed(0)
torchcudnn.benchmark = True
torchcudnn.enabled = True
torchcudnn.deterministic = True


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.to_pil = transforms.ToPILImage()
        pprint(self.args)

        self.path = construct_path_dict(proj_root=proj_root, exp_name=user_args.ex_name)  # self.args["Experiment_name"])
        pre_mkdir(path_config=self.path)
        
        # backup used file
        shutil.copy(f"{proj_root}/config.py", self.path["cfg_log"])
        shutil.copy(f"{proj_root}/train.py", self.path["trainer_log"])
        shutil.copy(f"{proj_root}/data/OBdataset.py", self.path["dataset_log"])
        shutil.copy(f"{proj_root}/network/ObPlaNet_simple.py", self.path["network_log"])
    
        # training data loader
        self.tr_loader = create_loader(
            self.args["train_data_path"], self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], 'train', self.args["batch_size"], self.args["num_workers"], True,
        )

        # load model
        self.dev = torch.device(f'cuda:{arg_config["gpu_id"]}')
        self.net = getattr(network, self.args["model"])(pretrained=True).to(self.dev)
        
        # loss functions
        self.loss = CrossEntropyLoss(ignore_index=255, reduction=self.args["reduction"]).to(self.dev)

        # optimizer
        self.opti = self.make_optim()

        # record loss
        tb_logger.configure(self.path['pth_log'], flush_secs=5)

        self.end_epoch = self.args["epoch_num"]
        if user_args.resume:
            try:
                self.resume_checkpoint(load_path=self.path["final_full_net"], mode="all")
            except:
                print(f"{self.path['final_full_net']} does not exist and we will load {self.path['final_state_net']}")
                self.resume_checkpoint(load_path=self.path["final_state_net"], mode="onlynet")
                self.start_epoch = self.end_epoch
        else:
            self.start_epoch = 0
        self.iter_num = self.end_epoch * len(self.tr_loader)


    def train(self):

        for curr_epoch in range(self.start_epoch, self.end_epoch):
            self.net.train()
            train_loss_record = AvgMeter()
            mimicking_loss_record = AvgMeter()

            # change learning rate
            if self.args["lr_type"] == "poly":
                self.change_lr(curr_epoch)
            elif self.args["lr_type"] == "decay":
                self.change_lr(curr_epoch)
            elif self.args["lr_type"] == "all_decay":
                self.change_lr(curr_epoch)
            else:
                raise NotImplementedError
            
            for train_batch_id, train_data in enumerate(self.tr_loader):
                curr_iter = curr_epoch * len(self.tr_loader) + train_batch_id

                self.opti.zero_grad()

                _, train_bgs, train_masks, train_fgs, train_targets, num, composite_list, feature_pos, _, _, _ = train_data
         
                train_bgs = train_bgs.to(self.dev, non_blocking=True)
                train_masks = train_masks.to(self.dev, non_blocking=True)
                train_fgs = train_fgs.to(self.dev, non_blocking=True)
                train_targets = train_targets.to(self.dev, non_blocking=True)
                num = num.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)
                
                # model training
                train_outs, feature_map = self.net(train_bgs, train_fgs, train_masks, 'train')
                
                mimicking_loss = feature_mimicking(composite_list, feature_pos, feature_map, num, self.dev)
                out_loss = self.loss(train_outs, train_targets.long())
                train_loss = out_loss + user_args.alpha*mimicking_loss 
                train_loss.backward()
                self.opti.step()

                train_iter_loss = out_loss.item()
                mimicking_iter_loss = mimicking_loss.item()
                train_batch_size = train_bgs.size(0)
                train_loss_record.update(train_iter_loss, train_batch_size)
                mimicking_loss_record.update(mimicking_iter_loss, train_batch_size)

                tb_logger.log_value('loss', train_loss.item(), step=self.net.Eiters)
        
                if self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0:
                    log = (
                        f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                        f"(L2)[Avg:{train_loss_record.avg:.3f}|Cur:{train_iter_loss:.3f}]"
                        f"(Lm)[Avg:{mimicking_loss_record.avg:.3f}][Cur:{mimicking_iter_loss:.3f}]"
                    )
                    print(log)
                    make_log(self.path["tr_log"], log)

            save_dir, save_name = os.path.split(self.path["final_full_net"])
            epoch_full_net_path = os.path.join(save_dir, str(curr_epoch + 1)+'_'+save_name)
            save_dir, save_name = os.path.split(self.path["final_state_net"])
            epoch_state_net_path = os.path.join(save_dir, str(curr_epoch + 1)+'_'+save_name) 
            
            self.save_checkpoint(curr_epoch + 1, full_net_path=epoch_full_net_path, state_net_path=epoch_state_net_path)
   

    def change_lr(self, curr):
        total_num = self.end_epoch
        if self.args["lr_type"] == "poly":
            ratio = pow((1 - float(curr) / total_num), self.args["lr_decay"])
            self.opti.param_groups[0]["lr"] = self.opti.param_groups[0]["lr"] * ratio
            self.opti.param_groups[1]["lr"] = self.opti.param_groups[0]["lr"]
        elif self.args["lr_type"] == "decay":
            ratio = 0.1
            if (curr % 9 == 0):
                self.opti.param_groups[0]["lr"] = self.opti.param_groups[0]["lr"] * ratio
                self.opti.param_groups[1]["lr"] = self.opti.param_groups[0]["lr"]
        elif self.args["lr_type"] == "all_decay":
            lr = self.args["lr"] * (0.5 ** (curr // 2))
            for param_group in self.opti.param_groups:
                param_group['lr'] = lr
        else:
            raise NotImplementedError

    def make_optim(self):
        if self.args["optim"] == "sgd_trick":
            params = [
                {
                    "params": [p for name, p in self.net.named_parameters() if ("bias" in name or "bn" in name)],
                    "weight_decay": 0,
                },
                {
                    "params": [
                        p for name, p in self.net.named_parameters() if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
            optimizer = SGD(
                params,
                lr=self.args["lr"],
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"],
            )
        elif self.args["optim"] == "f3_trick":
            backbone, head = [], []
            for name, params_tensor in self.net.named_parameters():
                if "encoder" in name:
                    backbone.append(params_tensor)
                else:
                    head.append(params_tensor)
            params = [
                {"params": backbone, "lr": 0.1 * self.args["lr"]},
                {"params": head, "lr": self.args["lr"]},
            ]
            optimizer = SGD(
                params=params,
                momentum=self.args["momentum"],
                weight_decay=self.args["weight_decay"],
                nesterov=self.args["nesterov"],
            )
        elif self.args["optim"] == "Adam_trick":
            optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args["lr"])
        else:
            raise NotImplementedError
        print("optimizer = ", optimizer)
        return optimizer

    def save_checkpoint(self, current_epoch, full_net_path, state_net_path):
        state_dict = {
            "epoch": current_epoch,
            "net_state": self.net.state_dict(),
            "opti_state": self.opti.state_dict(),
        }
        torch.save(state_dict, full_net_path)
        torch.save(self.net.state_dict(), state_net_path)

    def resume_checkpoint(self, load_path, mode="all"):
        """
        Args:
            load_path (str): path of pretrained model
            mode (str): 'all'：resume all information；'onlynet':only resume model parameters
        """
        if os.path.exists(load_path) and os.path.isfile(load_path):
            print(f" =>> loading checkpoint '{load_path}' <<== ")
            checkpoint = torch.load(load_path, map_location=self.dev)
            if mode == "all":
                self.start_epoch = checkpoint["epoch"]
                self.net.load_state_dict(checkpoint["net_state"])
                self.opti.load_state_dict(checkpoint["opti_state"])
                print(f" ==> loaded checkpoint '{load_path}' (epoch {checkpoint['epoch']})")
            elif mode == "onlynet":
                self.net.load_state_dict(checkpoint)
                print(f" ==> loaded checkpoint '{load_path}' " f"(only has the net's weight params) <<== ")
            else:
                raise NotImplementedError
        else:
            raise Exception(f"{load_path} is not correct.")


def feature_mimicking(composites, feature_pos, feature_map, num, device):

    net_ = pretrained_resnet18_4ch(pretrained=True).to(device)

    composite_cat_list = []
    pos_feature = torch.zeros(int(num.sum()), 512, 1, 1).to(device)
    count = 0
    for i in range(num.shape[0]):
        composite_cat_list.append(composites[i, :num[i], :, :, :])
        for j in range(num[i]):
            pos_feature[count, :, 0, 0] = feature_map[i, :, int(feature_pos[i, j, 1]), int(feature_pos[i, j, 0])]
            count += 1
    composites_ = torch.cat(composite_cat_list, dim=0)
    composite_feature = net_(composites_)
    composite_feature = nn.AdaptiveAvgPool2d(1)(composite_feature)  
    pos_feature.view(-1, 512)
    composite_feature.view(-1, 512)
    
    mimicking_loss_criter = nn.MSELoss()
    mimicking_loss = mimicking_loss_criter(pos_feature, composite_feature)

    return mimicking_loss 


if __name__ == "__main__":
    trainer = Trainer(arg_config)
    print(f" ===========>> {datetime.now()}: begin training <<=========== ")
    trainer.train()
    print(f" ===========>> {datetime.now()}: end training <<=========== ")

