import os
import csv
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
from config import arg_config

fg_scale_num = 8
save_img_flag = True

   
def collect_info(json_file, bg_dir, fg_dir):

    f_json = json.load(open(json_file, 'r'))
    return [
        (                               
            row['imgID'], row['annID'], row['scID'], 
            os.path.join(bg_dir, "%012d.jpg" % int(row['scID'])),        
            os.path.join(fg_dir, "foreground/{}.jpg".format(int(row['annID']))),
            os.path.join(fg_dir, "foreground/mask_{}.jpg".format(int(row['annID'])))
        )
        for _, row in enumerate(f_json)
    ]
    

fg_scales = list(range(1, fg_scale_num+1))
fg_scales = [i/(1+fg_scale_num+1) for i in fg_scales]
    
fg_bg_dict = dict()
args = arg_config 
data = collect_info(args["test_data_path"], args["bg_dir"], args["fg_dir"])

csv_dir = './data/data'
scaled_fg_dir = f'./data/data/fg/test_{fg_scale_num}scales/'
scaled_mask_dir = f'./data/data/mask/test_{fg_scale_num}scales/'

os.makedirs(scaled_fg_dir, exist_ok=True)
os.makedirs(scaled_mask_dir, exist_ok=True)

csv_file =  os.path.join(csv_dir, f'test_data_{fg_scale_num}scales.csv')
json_file = csv_file.replace('.csv', '.json')

file = open(csv_file, mode='w', newline='')
writer = csv.writer(file)


csv_head = ['imgID', 'annID', 'scID', 'scale', 'newWidth', 'newHeight', 'pos_label', 'neg_label']
writer.writerow(csv_head)



for _,index in enumerate(tqdm(range(len(data)))):
    imgID, fg_id, bg_id, bg_path, fg_path, mask_path = data[index] 
    if (fg_id, bg_id) in fg_bg_dict.keys():
        continue
    fg_bg_dict[(fg_id, bg_id)] = 1
    
    
    bg_img = Image.open(bg_path)  
    if len(bg_img.split()) != 3: 
        bg_img = bg_img.convert("RGB")
    bg_img_aspect =  bg_img.height/bg_img.width
    fg_tocp = Image.open(fg_path).convert("RGB")
    mask_tocp = Image.open(mask_path).convert("RGB")
    fg_tocp_aspect = fg_tocp.height/fg_tocp.width
    
    for fg_scale in fg_scales:
        if fg_tocp_aspect>bg_img_aspect:
            new_height = bg_img.height*fg_scale
            new_width = new_height/fg_tocp.height*fg_tocp.width
        else:
            new_width = bg_img.width*fg_scale
            new_height = new_width/fg_tocp.width*fg_tocp.height
        
        new_height = int(new_height)
        new_width = int(new_width) 
        
        if save_img_flag:
            top = int((bg_img.height-new_height)/2)
            bottom = top+new_height
            left = int((bg_img.width-new_width)/2)
            right = left+new_width
     
            fg_img_ = fg_tocp.resize((new_width, new_height))
            mask_ = mask_tocp.resize((new_width, new_height))
            
            fg_img_ = np.array(fg_img_)       
            mask_ = np.array(mask_)
            
            fg_img = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
            mask  = np.zeros((bg_img.height, bg_img.width, 3), dtype=np.uint8) 
            
            fg_img[top:bottom, left:right, :] = fg_img_
            mask[top:bottom, left:right, :] = mask_
            
            fg_img = Image.fromarray(fg_img.astype(np.uint8))
            mask = Image.fromarray(mask.astype(np.uint8))
            
            basename = f'{fg_id}_{bg_id}_{new_width}_{new_height}.jpg'
            fg_img_path = os.path.join(scaled_fg_dir, basename)
            mask_path = os.path.join(scaled_mask_dir, basename)
            fg_img.save(fg_img_path)
            mask.save(mask_path)
        
        writer.writerow([imgID, fg_id, bg_id, fg_scale, new_width, new_height, None, None])
        
        
file.close()  
            
# convert csv file to json file            
csv_data = []
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['pos_label']=="":
            row['pos_label'] = [[0,0]]
        if row['neg_label']=="":
            row['neg_label'] = [[0,0]]
        csv_data.append(row)

with open(json_file, mode='w') as file:
    json.dump(csv_data, file, indent=4)

