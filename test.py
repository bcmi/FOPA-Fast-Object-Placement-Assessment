import os
import warnings
from pprint import pprint
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

with warnings.catch_warnings():  # 过滤警告
    warnings.filterwarnings("ignore", category=FutureWarning)
import argparse

import network
from config import arg_config
from data.OBdataset import create_loader


class Evaluator:
    def __init__(self, args, checkpoint_path):
        super(Evaluator, self).__init__()
        self.args = args
        self.dev = torch.device("cuda:0")
        self.to_pil = transforms.ToPILImage()
        pprint(self.args)

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
        print('load pretrained weights from ', checkpoint_path)
        self.net = getattr(network, self.args["model"])(
            pretrained=False).to(self.dev)
        self.net.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.net = self.net.to(self.dev).eval()

    def evalutate_model(self):
        '''
        used to check the f1 and bacc metric of specified model
        '''
        correct = torch.zeros(1).squeeze().to(self.dev, non_blocking=True)
        total = torch.zeros(1).squeeze().to(self.dev, non_blocking=True)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        with torch.no_grad():
            for batch_index, test_data in enumerate(tqdm(self.ts_loader)):

                index, test_bgs, test_masks, test_fgs, log_depth, semantic_map, test_targets, nums, composite_list, feature_pos, w, h, savename = test_data
                test_bgs = test_bgs.to(self.dev, non_blocking=True)
                test_masks = test_masks.to(self.dev, non_blocking=True)
                test_fgs = test_fgs.to(self.dev, non_blocking=True)
                nums = nums.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)

                test_outs, feature_map = self.net(
                    test_bgs, test_fgs, test_masks, 'val')
                test_preds = np.argmax(test_outs.cpu().numpy(), axis=1)
                test_targets = test_targets.cpu().numpy()

                TP += ((test_preds == 1) & (test_targets == 1)).sum()
                TN += ((test_preds == 0) & (test_targets == 0)).sum()
                FP += ((test_preds == 1) & (test_targets == 0)).sum()
                FN += ((test_preds == 0) & (test_targets == 1)).sum()

                correct += (test_preds == test_targets).sum()
                total += nums.sum()

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            fscore = (2 * precision * recall) / (precision + recall)
            fscore_str = 'F-1 Measure: %f, ' % fscore

            weighted_acc = (TP / (TP + FN) + TN / (TN + FP)) * 0.5
            weighted_acc_str = 'Weighted acc measure: %f, ' % weighted_acc
            pred_neg = TN / (TN + FP)
            pred_pos = TP / (TP + FN)
            pred_str = 'pred_neg: %f, pred_pos: %f ,' % (pred_neg, pred_pos)

            print(fscore_str)
            print(weighted_acc_str)

    def get_heatmap(self, mode):
        '''
        used to get the heatmap for the test dataset
        '''
        assert mode == 'train' or mode == 'test'

        save_root = f"heatmap/{mode}"
        if not os.path.exists(save_root):
            print(f"Create directory {save_root}")
            os.makedirs(save_root)

        if mode == 'train':
            self.data_loader = create_loader(
                self.args["tr_data_path"], self.args["coco_dir"], self.args["fg_dir"], self.args["mask_dir"],
                self.args["input_size"], 'train', 1, self.args["num_workers"], False,
            )
        elif mode == 'test':
            self.data_loader = create_loader(
                self.args["ts_data_path"], self.args["coco_dir"], self.args["fg_dir"], self.args["mask_dir"],
                self.args["input_size"], 'test', 1, self.args["num_workers"], False,
            )      

        m = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for batch_index, test_data in enumerate(tqdm(self.data_loader)):
                index, test_bgs, test_masks, test_fgs, log_depth, semantic_map, test_targets, nums, composite_list, feature_pos, w, h, savename = test_data
                test_bgs = test_bgs.to(self.dev, non_blocking=True)
                test_masks = test_masks.to(self.dev, non_blocking=True)
                test_fgs = test_fgs.to(self.dev, non_blocking=True)
                nums = nums.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)

                test_outs, feature_map = self.net(
                    test_bgs, test_fgs, test_masks, 'val')
                
                test_outs = m(test_outs)
                # 生成 heatmap
                test_outs = test_outs[:,1,:,:] # b(=1),256,256
                test_outs = transforms.ToPILImage()(test_outs)
                test_outs.save(os.path.join(save_root, savename[0]))

if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default= "test model")
    parser.add_argument('--path', type=str, default= "./best_weight.pth")
    args = parser.parse_args()

    evaluator = Evaluator(arg_config, checkpoint_path=args.path)
    if args.mode== "test model":
        evaluator.evalutate_model() 
    else:
        evaluator.get_heatmap(args.mode)  

