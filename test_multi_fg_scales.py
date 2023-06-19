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

    def get_heatmap_multi_scales(self, fg_scale_num):
        '''
        generate heatmap for each pair of scaled foreground and background  
        '''
        
        datatype= f"test_{fg_scale_num}scales"
        
        save_dir, base_name = os.path.split(self.checkpoint_path)
        heatmap_dir = os.path.join(save_dir, base_name.replace('.pth', f'_{datatype}_heatmap'))
        
        if not os.path.exists(heatmap_dir):
            print(f"Create directory {heatmap_dir}")
            os.makedirs(heatmap_dir)     

        
        json_path = os.path.join('./data/data', f"test_data_{fg_scale_num}scales.json")  
        
        self.ts_loader = create_loader(
            json_path, self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"],
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
        
    def generate_composite_multi_scales(self, fg_scale_num, composite_num):
        '''
        generate composite images for each pair of scaled foreground and background 
        '''
        
        fg_scales = list(range(1, fg_scale_num+1))
        fg_scales = [i/(1+fg_scale_num+1) for i in fg_scales]

        icount = 0
    
        save_dir, base_name = os.path.split(self.checkpoint_path)
        heatmap_dir = os.path.join(save_dir, base_name.replace('.pth', f'_test_{fg_scale_num}scales_heatmap'))
        if not os.path.exists(heatmap_dir):
            print(f"{heatmap_dir} does not exist! Please first use 'heatmap' mode to generate heatmaps")
                          
        json_path = os.path.join('./data/data', f"test_data_{fg_scale_num}scales.json")  
                          
        data = _collect_info(json_path, self.args["bg_dir"], self.args["fg_dir"], self.args["mask_dir"], 'test')
        for index in range(len(data)):
            _, _, bg_path, fg_path, _, scale, _, _, fg_path_2, mask_path_2, w, h = data[index] 
          
            fg_name =  fg_path.split('/')[-1][:-4]
            save_name = fg_name + '_' + str(scale) 
            segs = fg_name.split('_')
            fg_id, bg_id = segs[0], segs[1] 
            if icount==0:

                bg_img = Image.open(bg_path)  
                if len(bg_img.split()) != 3:  
                    bg_img = bg_img.convert("RGB")
                fg_tocp = Image.open(fg_path_2).convert("RGB")
                mask_tocp = Image.open(mask_path_2).convert("RGB")
                         
                composite_dir = os.path.join(save_dir, base_name.replace('.pth', f'_test_{fg_scale_num}scales_composite'), f'{fg_id}_{bg_id}')
                if not os.path.exists(composite_dir):
                    print(f"Create directory {composite_dir}")
                    os.makedirs(composite_dir)
                                            
                heatmap_center_list = []
                fg_size_list = []
            
            icount += 1
            heatmap = Image.open(os.path.join(heatmap_dir, save_name+'.jpg'))
            heatmap = np.array(heatmap)
            # exclude boundary
            heatmap_center = np.zeros_like(heatmap, dtype=np.float_)
            hb= int(h/bg_img.height*heatmap.shape[0]/2)
            wb = int(w/bg_img.width*heatmap.shape[1]/2)
            heatmap_center[hb:-hb, wb:-wb] = heatmap[hb:-hb, wb:-wb]
            heatmap_center_list.append(heatmap_center)
            fg_size_list.append((h,w))
            
            if icount==fg_scale_num:
                icount = 0
                heatmap_center_stack = np.stack(heatmap_center_list)
                # sort pixels in a descending order based on the heatmap 
                sorted_indices = np.argsort(-heatmap_center_stack, axis=None)
                sorted_indices = np.unravel_index(sorted_indices, heatmap_center_stack.shape)
                for i in range(composite_num):
                    iscale, y_, x_ = sorted_indices[0][i], sorted_indices[1][i], sorted_indices[2][i]
                    h, w = fg_size_list[iscale]
                    x_ = x_/heatmap.shape[1]*bg_img.width
                    y_ = y_/heatmap.shape[0]*bg_img.height
                    x = int(x_ - w / 2)
                    y = int(y_ - h / 2)
                    # make composite image with foreground, background, and placement 
                    composite_img, composite_msk = make_composite_PIL(fg_tocp, mask_tocp, bg_img, [x, y, w, h], return_mask=True)
                    save_img_path = os.path.join(composite_dir, f'{fg_id}_{bg_id}_{x}_{y}_{w}_{h}.jpg')
                    save_msk_path = os.path.join(composite_dir, f'{fg_id}_{bg_id}_{x}_{y}_{w}_{h}.png')
                    composite_img.save(save_img_path)
                    composite_msk.save(save_msk_path)
                    print(save_img_path)

     

if __name__ == "__main__":
    print("cuda: ", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default= "composite") 
    parser.add_argument('--path', type=str, default= "demo2023-05-19-22:36:47.952468")
    parser.add_argument('--epoch', type=int, default= 20)
    args = parser.parse_args()
    
    fg_scale_num = 8
    composite_num = 50

    full_path = os.path.join('output', args.path, 'pth', f'{args.epoch}_state_final.pth')

    if not os.path.exists(full_path):
        print(f'{full_path} does not exist!')
    else:
        evaluator = Evaluator(arg_config, checkpoint_path=full_path) 
        if args.mode== "heatmap":
            evaluator.get_heatmap_multi_scales(fg_scale_num)
        elif args.mode== "composite":
            evaluator.generate_composite_multi_scales(fg_scale_num, composite_num) 

