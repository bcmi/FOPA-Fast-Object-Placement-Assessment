# @Time    : 2020/7/8
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : config.py
# @Project : HDFNet
# @GitHub  : https://github.com/lartpang
import os

__all__ = ["proj_root", "arg_config"]

proj_root = os.path.dirname(__file__)
datasets_root = "./data/data"

tr_data_path = os.path.join(datasets_root, "train_pair_new.json")  # 训练集数据路径
ts_data_path = os.path.join(datasets_root, "test_pair_new.json")  # 测试集数据路径
coco_dir = './data/data/train2017'
fg_dir = os.path.join(datasets_root, "fg")  # 合成图中前景图片路径
mask_dir = os.path.join(datasets_root, "mask")  # 合成图中前景mask路径

# 配置区域 #####################################################################
arg_config = {
    # 常用配置
    "model": "ObPlaNet_resnet18",  # 模型名称, 修改后注意修改network__init__文件夹下的import
    "suffix": "simple_mask_adam", # 后缀
    "resume": False,  # 是否需要恢复模型
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 25,
    "lr": 0.0005,
    "tr_data_path": tr_data_path,
    "ts_data_path": ts_data_path,
    "coco_dir": coco_dir,
    "fg_dir": fg_dir,
    "mask_dir": mask_dir,

    "print_freq": 10,  # >0, 保存迭代过程中的信息
    "prefix": (".jpg", ".png"),
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    "optim": "Adam_trick",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "nesterov": False,
    "lr_type": "all_decay",  # 学习率调整的策略
    "lr_decay": 0.9,  # poly
    "batch_size": 4,
    "num_workers": 6,
    "input_size": 256,  # 输入图片大小
    #"depth_loss":0,
    #"semantic_loss":0,
    "gpu_id": 3,
    "ex_name":"demo", #实验名称，开始新实验后修改
    "Experiment_name": "Model2_multiscale_fix_fm_alpha_test",
}
