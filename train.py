# -*- coding: utf-8 -*-
# @Time    : 2020/7/5
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : main.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os
import os.path as osp
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import torch.backends.cudnn as torchcudnn
from PIL import Image
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, DataParallel
from torch.optim import SGD, Adam
from torchvision import transforms
from tqdm import tqdm

import warnings

with warnings.catch_warnings(): # 过滤警告
    warnings.filterwarnings("ignore", category=FutureWarning)
import tensorboard_logger as tb_logger

import network
from config import arg_config, proj_root
from data.OBdataset import create_loader

from utils.misc import AvgMeter, construct_path_dict, make_log, pre_mkdir#, CrossEntropyLoss2d
from utils.metric import CalTotalMetric

from backbone.ResNet import pretrained_resnet18_4ch
import torch.nn as nn
import random
import argparse

parser = argparse.ArgumentParser(description='Model2_multiscale_fix_fm_alpha_test')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size',
                    choices=[1, 3, 5, 7])
parser.add_argument('--multi_scale', type=int, default=2, help='kernel size',
                    choices=[1, 2, 3, 4, 5])
parser.add_argument('--ex_name', type=str, default=arg_config["ex_name"])
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

args_2 = parser.parse_args()


# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
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

        # 设置日志文件
        if self.args["suffix"]:
            self.model_name = self.args["model"] + "_" + self.args["suffix"]
        else:
            self.model_name = self.args["model"]
        self.path = construct_path_dict(proj_root=proj_root, exp_name=args_2.ex_name)  # self.args["Experiment_name"])

        pre_mkdir(path_config=self.path)
        # backup used file
        shutil.copy(f"{proj_root}/config.py", self.path["cfg_log"])
        shutil.copy(f"{proj_root}/train.py", self.path["trainer_log"])
        shutil.copy(f"{proj_root}/data/OBdataset.py", self.path["dataset_log"])
        shutil.copy(f"{proj_root}/network/ObPlaNet_simple.py", self.path["network_log"])
        
        self.save_path = self.path["save"]
        self.save_pre = self.args["save_pre"]
        self.bestF1 = 0.

        # 加载数据集
        self.tr_loader = create_loader(
            self.args["tr_data_path"], self.args["coco_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], 'train', self.args["batch_size"], self.args["num_workers"], True,
        )
        self.ts_loader = create_loader(
            self.args["ts_data_path"], self.args["coco_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], 'test', self.args["batch_size"], self.args["num_workers"], False,
        )

        # 加载model
        self.dev = torch.device(f'cuda:{arg_config["gpu_id"]}')# if torch.cuda.is_available() else "cpu"):
        self.net = getattr(network, self.args["model"])(pretrained=True).to(self.dev)
        print("scale:",self.net.scale)
        
        # 损失函数
        self.loss = CrossEntropyLoss(ignore_index=255, reduction=self.args["reduction"]).to(self.dev)
        #self.depth_loss = MSELoss()
        #self.semantic_loss = CrossEntropyLoss(ignore_index=255, reduction=self.args["reduction"]).to(self.dev)
        # 设置优化器
        self.opti = self.make_optim()

        # 记录LOSS
        tb_logger.configure(self.path['pth_log'], flush_secs=5)

        # 训练相关
        self.end_epoch = self.args["epoch_num"]
        if self.args["resume"]:
            try:
                self.resume_checkpoint(load_path=self.path["final_full_net"], mode="all")
            except:
                print(f"{self.path['final_full_net']} does not exist and we will load {self.path['final_state_net']}")
                self.resume_checkpoint(load_path=self.path["final_state_net"], mode="onlynet")
                self.start_epoch = self.end_epoch
        else:
            self.start_epoch = 0
        self.iter_num = self.end_epoch * len(self.tr_loader)

    def total_loss(self, train_preds, train_alphas):
        loss_list = []
        loss_item_list = []

        assert len(self.loss_funcs) != 0, "请指定损失函数`self.loss_funcs`"
        for loss in self.loss_funcs:
            loss_out = loss(train_preds, train_alphas)
            loss_list.append(loss_out)
            loss_item_list.append(f"{loss_out.item():.5f}")

        train_loss = sum(loss_list)
        return train_loss, loss_item_list

    def train(self):
        """训练模型"""
        for curr_epoch in range(self.start_epoch, self.end_epoch):
            self.net.train()
            train_loss_record = AvgMeter()
            mimicking_loss_record = AvgMeter()

            # 改变学习率
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

                ###加载数据
                index, train_bgs, train_masks, train_fgs, log_depth, semantic_map, train_targets, num, composite_list, feature_pos, w, h = train_data
         
                train_bgs = train_bgs.to(self.dev, non_blocking=True)
                train_masks = train_masks.to(self.dev, non_blocking=True)
                train_fgs = train_fgs.to(self.dev, non_blocking=True)
                train_targets = train_targets.to(self.dev, non_blocking=True)
                num = num.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)
                log_depth = log_depth.to(self.dev, non_blocking=True) 
                semantic_map = semantic_map.to(self.dev, non_blocking=True)
                semantic_map = semantic_map.long()
                
                # 送入模型训练
                train_outs, feature_map = self.net(train_bgs, train_fgs, train_masks, 'train')
                
                mimicking_loss = feature_mimicking(composite_list, feature_pos, feature_map, num, self.dev)
                out_loss = self.loss(train_outs, train_targets.long())
                #semantic_loss = self.semantic_loss(seg_feature, semantic_map)
                #depth_loss = self.depth_loss(pre_log_depth, log_depth)
                #depth_loss = 0
                train_loss = out_loss + mimicking_loss #+ self.args["semantic_loss"] * semantic_loss #+ self.args["depth_loss"] * depth_loss
                train_loss.backward()
                self.opti.step()

                # 仅在累计的时候使用item()获取数据
                train_iter_loss = out_loss.item()
                mimicking_iter_loss = mimicking_loss.item()
                train_batch_size = train_bgs.size(0)
                train_loss_record.update(train_iter_loss, train_batch_size)
                mimicking_loss_record.update(mimicking_iter_loss, train_batch_size)

                tb_logger.log_value('loss', train_loss.item(), step=self.net.Eiters)

                # 记录每一次迭代的数据
                if self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0:
                    log = (
                        f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                        # f"[{self.model_name}]"
                        f"[Lr:{self.opti.param_groups[0]['lr']:.7f}]"
                        f"(L2)[Avg:{train_loss_record.avg:.3f}|Cur:{train_iter_loss:.3f}]"
                        # f"(Lm)[Avg:{mimicking_loss_record.avg:.3f}][Cur:{mimicking_iter_loss:.3f}]"
                    )
                    print(log)
                    make_log(self.path["tr_log"], log)

            # 每个周期都进行保存测试，保存的是针对第curr_epoch+1周期的参数
            self.save_checkpoint(
                curr_epoch + 1, full_net_path=self.path["final_full_net"], state_net_path=self.path["final_state_net"],
            )

            # 每个周期都对模型当前结果进行一次测试
            self.test(self.ts_loader)

        # 进行最终的测试，首先输出验证结果
        print(f" ==>> 训练结束 <<== ")

    def test(self, dataloader):
        """测试阶段"""
        self.net.eval()
        correct = torch.zeros(1).squeeze().to(self.dev, non_blocking=True)
        total = torch.zeros(1).squeeze().to(self.dev, non_blocking=True)
        tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        mimicking_loss_record = AvgMeter()
        for test_batch_id, test_data in tqdm_iter:
            self.net.eval()
            tqdm_iter.set_description(f"{self.model_name}:" f"te=>{test_batch_id + 1}")
            with torch.no_grad():
                # 加载数据
                index, test_bgs, test_masks, test_fgs, log_depth, semantic_map, test_targets, nums, composite_list, feature_pos, w, h = test_data
                test_bgs = test_bgs.to(self.dev, non_blocking=True)
                test_masks = test_masks.to(self.dev, non_blocking=True)
                test_fgs = test_fgs.to(self.dev, non_blocking=True)
                nums = nums.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)
                log_depth = log_depth.to(self.dev, non_blocking=True)
                semantic_map = semantic_map.to(self.dev, non_blocking=True) 

                test_outs, feature_map  = self.net(test_bgs, test_fgs, test_masks, 'val')
                test_preds = np.argmax(test_outs.cpu().numpy(), axis=1)
                test_targets = test_targets.cpu().numpy()

                mimicking_loss = feature_mimicking(composite_list, feature_pos, feature_map, nums, self.dev)
                mimicking_iter_loss = mimicking_loss.item()
                test_batch_size = test_bgs.size(0)
                mimicking_loss_record.update(mimicking_iter_loss, test_batch_size)

                TP += ((test_preds == 1) & (test_targets == 1)).sum()
                TN += ((test_preds == 0) & (test_targets == 0)).sum()
                FP += ((test_preds == 1) & (test_targets == 0)).sum()
                FN += ((test_preds == 0) & (test_targets == 1)).sum()

                correct += (test_preds == test_targets).sum()
                total += nums.sum()

        # make_log(self.path["tr_log"], acc_str)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore_str = 'F-1 Measure: %f, ' % fscore
        if fscore > self.bestF1:
            self.bestF1 = fscore
            weight_path = os.path.join('output', args_2.ex_name, 'pth', 'best_weight.pth')
            torch.save(self.net.state_dict(), weight_path)
        weighted_acc = (TP / (TP + FN) + TN / (TN + FP)) * 0.5
        weighted_acc_str = 'Weighted acc measure: %f, ' % weighted_acc
        pred_neg = TN / (TN + FP)
        pred_pos = TP / (TP + FN)
        pred_str = 'pred_neg: %f, pred_pos: %f ,' % (pred_neg, pred_pos)
        log = fscore_str + weighted_acc_str + pred_str + 'TP: %f, TN: %f, FP: %f, FN: %f' % (TP, TN, FP, FN)

        log += ' [LM(test-L2): %f]' % mimicking_loss_record.avg

        print(log)
        make_log(self.path["tr_log"], log)
        make_log(self.path["te_log"], log)

    def change_lr(self, curr):
        """
        更改学习率

        Args:
            curr (int): 当前周期
        """
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
        """初始化optimizer"""
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
        """
        保存完整参数模型（大）和状态参数模型（小）

        Args:
            current_epoch (int): 当前周期
            full_net_path (str): 保存完整参数模型的路径
            state_net_path (str): 保存模型权重参数的路径
        """
        state_dict = {
            "epoch": current_epoch,
            "net_state": self.net.state_dict(),
            "opti_state": self.opti.state_dict(),
        }
        torch.save(state_dict, full_net_path)
        torch.save(self.net.state_dict(), state_net_path)

    def resume_checkpoint(self, load_path, mode="all"):
        """
        从保存节点恢复模型

        Args:
            load_path (str): 模型存放路径
            mode (str): 选择哪种模型恢复模式：'all'：回复完整模型，包括训练中的的参数；'onlynet'：仅恢复模型权重参数
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
            raise Exception(f"{load_path}路径不正常，请检查")


def feature_mimicking(composites, feature_pos, feature_map, num, device):
    alpha = 16 #损失权重
    net_ = pretrained_resnet18_4ch(pretrained=True).to(device)
    # features = list(net.children())[:-1]
    # net_ = nn.Sequential(*features)
    # net_ = nn.Sequential(*list(net.children()[:-2]))
    composite_cat_list = []
    pos_feature = torch.zeros(int(num.sum()), 512, 1, 1).to(device)
    sum = 0
    for i in range(num.shape[0]):
        composite_cat_list.append(composites[i, :num[i], :, :, :])
        for j in range(num[i]):
            pos_feature[sum, :, 0, 0] = feature_map[i, :, int(feature_pos[i, j, 1]), int(feature_pos[i, j, 0])]
            sum += 1
    composites_ = torch.cat(composite_cat_list, dim=0)
    composite_feature = net_(composites_)
    composite_feature = nn.AdaptiveAvgPool2d(1)(composite_feature)  # pos_num(8),512,1,1
    pos_feature.view(-1, 512)
    composite_feature.view(-1, 512)
    # L2损失
    mimicking_loss_criter = nn.MSELoss()
    mimicking_loss = mimicking_loss_criter(pos_feature, composite_feature)
    # mimicking_loss = torch.zeros(1).to(device)
    # for i in range(num.sum()):
    #     similarity = torch.cosine_similarity(pos_feature[i], composite_feature[i], dim=0)
    #     mimicking_loss += 1 - similarity.squeeze(0)
    return mimicking_loss * alpha


if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    trainer = Trainer(arg_config)
    print(f" ===========>> {datetime.now()}: 开始训练 <<=========== ")
    trainer.train()
    # trainer.test()
    print(f" ===========>> {datetime.now()}: 结束训练 <<=========== ")

