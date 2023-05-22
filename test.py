import os
import argparse
import torch
import numpy as np
from PIL import Image
from pprint import pprint
from torchvision import transforms
from tqdm import tqdm

import network
from config import arg_config
from data.OBdataset import create_loader, _collect_info
from data.OBdataset import make_composite_PIL


class Evaluator:
    def __init__(self, args, checkpoint_path):
        super(Evaluator, self).__init__()
        self.args = args
        self.dev = torch.device("cuda:0")
        self.to_pil = transforms.ToPILImage()
        self.checkpoint_path = checkpoint_path
        pprint(self.args)

        print('load pretrained weights from ', checkpoint_path)
        self.net = getattr(network, self.args["model"])(
            pretrained=False).to(self.dev)
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.dev), strict=False)
        self.net = self.net.to(self.dev).eval()
        self.softmax = torch.nn.Softmax(dim=1)

    def evalutate_model(self, datatype):
        '''
        calculate F1 and bAcc metrics
        '''
    
        correct = 0
        total = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        assert datatype=='train' or datatype=='test'
   
        self.ts_loader = create_loader(
            self.args[f"{datatype}_data_path"], self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], datatype, self.args["batch_size"], self.args["num_workers"], False,
        )

        with torch.no_grad():
            
            for _, test_data in enumerate(tqdm(self.ts_loader)):
                _, test_bgs, test_masks, test_fgs, test_targets, nums, composite_list, feature_pos, _, _, _ = test_data
                test_bgs = test_bgs.to(self.dev, non_blocking=True)
                test_masks = test_masks.to(self.dev, non_blocking=True)
                test_fgs = test_fgs.to(self.dev, non_blocking=True)
                nums = nums.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)

                test_outs, _ = self.net(test_bgs, test_fgs, test_masks, 'val')
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
            weighted_acc = (TP / (TP + FN) + TN / (TN + FP)) * 0.5

            print('F-1 Measure: %f, ' % fscore)
            print('Weighted acc measure: %f, ' % weighted_acc)

    def get_heatmap(self, datatype):
        '''
        generate heatmap for each pair of scaled foreground and background  
        '''
        
        save_dir, base_name = os.path.split(self.checkpoint_path)
        heatmap_dir = os.path.join(save_dir, base_name.replace('.pth', f'_{datatype}_heatmap'))
        
        if not os.path.exists(heatmap_dir):
            print(f"Create directory {heatmap_dir}")
            os.makedirs(heatmap_dir)     

        
        
        self.ts_loader = create_loader(
            self.args[f"{datatype}_data_path"], self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"],
            self.args["input_size"], datatype, 1, self.args["num_workers"], False,
        )
        
        with torch.no_grad():
            for _, test_data in enumerate(tqdm(self.ts_loader)):
                _, test_bgs, test_masks, test_fgs, _, nums, composite_list, feature_pos, _, _, save_name = test_data
                test_bgs = test_bgs.to(self.dev, non_blocking=True)
                test_masks = test_masks.to(self.dev, non_blocking=True)
                test_fgs = test_fgs.to(self.dev, non_blocking=True)
                nums = nums.to(self.dev, non_blocking=True)
                composite_list = composite_list.to(self.dev, non_blocking=True)
                feature_pos = feature_pos.to(self.dev, non_blocking=True)

                test_outs, _ = self.net(test_bgs, test_fgs, test_masks, 'test')    
                test_outs = self.softmax(test_outs)
           
                test_outs = test_outs[:,1,:,:] 
                test_outs = transforms.ToPILImage()(test_outs)
                test_outs.save(os.path.join(heatmap_dir, save_name[0]))
                
    def generate_composite(self, datatype, composite_num):
        '''
        generate composite images for each pair of scaled foreground and background 
        '''
        
        save_dir, base_name = os.path.split(self.checkpoint_path)
        heatmap_dir = os.path.join(save_dir, base_name.replace('.pth', f'_{datatype}_heatmap'))
        if not os.path.exists(heatmap_dir):
            print(f"{heatmap_dir} does not exist! Please first use 'heatmap' mode to generate heatmaps")
                          
        data = _collect_info(self.args[f"{datatype}_data_path"], self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"], 'test')
        for index in range(len(data)):
            _, _, bg_path, fg_path, _, scale, _, _, fg_path_2, mask_path_2, w, h = data[index] 
          
            fg_name =  fg_path.split('/')[-1][:-4]
            save_name = fg_name + '_' + str(scale) 
    
            bg_img = Image.open(bg_path)  
            if len(bg_img.split()) != 3:  
                bg_img = bg_img.convert("RGB")
            fg_tocp = Image.open(fg_path_2).convert("RGB")
            mask_tocp = Image.open(mask_path_2).convert("RGB")
                         
            composite_dir = os.path.join(save_dir, base_name.replace('.pth', f'_{datatype}_composite'), save_name)
            if not os.path.exists(composite_dir):
                print(f"Create directory {composite_dir}")
                os.makedirs(composite_dir)                   
            
            heatmap = Image.open(os.path.join(heatmap_dir, save_name+'.jpg'))
            heatmap = np.array(heatmap)
            
            # exclude boundary
            heatmap_center = np.zeros_like(heatmap, dtype=np.float_)
            hb= int(h/bg_img.height*heatmap.shape[0]/2)
            wb = int(w/bg_img.width*heatmap.shape[1]/2)
            heatmap_center[hb:-hb, wb:-wb] = heatmap[hb:-hb, wb:-wb]
            
            # sort pixels in a descending order based on the heatmap 
            sorted_indices = np.argsort(-heatmap_center, axis=None)
            sorted_indices = np.unravel_index(sorted_indices, heatmap_center.shape)
            for i in range(composite_num):
                y_, x_ = sorted_indices[0][i], sorted_indices[1][i]
                x_ = x_/heatmap.shape[1]*bg_img.width
                y_ = y_/heatmap.shape[0]*bg_img.height
                x = int(x_ - w / 2)
                y = int(y_ - h / 2)
                # make composite image with foreground, background, and placement 
                composite_img = make_composite_PIL(fg_tocp, mask_tocp, bg_img, [x, y, w, h])
                save_img_path = os.path.join(composite_dir, f'{save_name}_{int(x_)}_{int(y_)}.jpg')
                composite_img.save(save_img_path)
                print(save_img_path)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    # "evaluate": calculate F1 and bAcc
    # "heatmap": generate FOPA heatmap
    # "composite": generate composite images based on the heatmap
    parser.add_argument('--mode', type=str, default= "composite") 
    # datatype: "train" or "test"
    parser.add_argument('--datatype', type=str, default= "test") 
    parser.add_argument('--path', type=str, default= "demo2023-05-19-22:36:47.952468")
    parser.add_argument('--epoch', type=int, default= 23)
    args = parser.parse_args()

    #full_path = os.path.join('output', args.path, 'pth', f'{args.epoch}_state_final.pth')
    full_path = 'best_weight.pth'  
    
    if not os.path.exists(full_path):
        print(f'{full_path} does not exist!')
    else:
        evaluator = Evaluator(arg_config, checkpoint_path=full_path)
        if args.mode== "evaluate":
            evaluator.evalutate_model(args.datatype) 
        elif args.mode== "heatmap":
            evaluator.get_heatmap(args.datatype)
        elif args.mode== "composite":
            evaluator.generate_composite(args.datatype, 50) 
        else:
            print(f'There is no {args.mode} mode.')    

