import os

__all__ = ["proj_root", "arg_config"]

proj_root = os.path.dirname(__file__)
datasets_root = "./data/data"

tr_data_path = os.path.join(datasets_root, "train_pair_new.json")  
ts_data_path = os.path.join(datasets_root, "test_pair_new.json")    

coco_dir = './data/data/train2017'
bg_dir = os.path.join(datasets_root, "bg")  
fg_dir = os.path.join(datasets_root, "fg")  
mask_dir = os.path.join(datasets_root, "mask") 

arg_config = {
  
    "model": "ObPlaNet_resnet18",  # model name
    "epoch_num": 25,
    "lr": 0.0005,
    "train_data_path": tr_data_path,
    "test_data_path": ts_data_path,
    "bg_dir": bg_dir,
    "fg_dir": fg_dir,
    "mask_dir": mask_dir,

    "print_freq": 10,  # >0, frequency of log print 
    "prefix": (".jpg", ".png"),
    "reduction": "mean",  # “mean” or “sum”
    "optim": "Adam_trick",  # optimizer
    "weight_decay": 5e-4,  # set as 0.0001 when finetuning
    "momentum": 0.9,
    "nesterov": False,
    "lr_type": "all_decay",  # learning rate schedule
    "lr_decay": 0.9,  # poly
    "batch_size": 8,
    "num_workers": 6,
    "input_size": 256,  # input size
    "gpu_id": 0,
    "ex_name":"demo", # experiment name
}
