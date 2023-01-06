import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import numpy as np
from scipy.linalg import hadamard 
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm 
torch.multiprocessing.set_sharing_strategy('file_system')
from network_patch_global import ResNet_DOLG
from utils.loss import OrthoHashLoss
from utils.tools import *
from utils.logger import setup_logger


def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[infer]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 50,
        "net": ResNet_DOLG,
        'S3_DIM':1024,
        "S4_DIM": 2048,
        'learn_cent':False,
        "dataset": "imagenet",
        "epoch": 500,
        "test_map": 2,
        "device": torch.device("cuda:6"),
        "bit_list": [64],
        'root_log': 'logs',
        'loss_type': 'arc', 
    }

    parser = argparse.ArgumentParser(description='OrthoHash')

    # loss related
    parser.add_argument('--scale', default=10, type=float, help='scale for cossim')
    parser.add_argument('--margin', default=0.15, type=float, help='ortho margin ')
    parser.add_argument('--margin-type', default='arc', choices=['cos', 'arc'], help='margin type')
    parser.add_argument('--ce', default=1.0, type=float, help='classification scale')
    parser.add_argument('--quan', default=1.0, type=float, help='quantization loss scale')
    parser.add_argument('--quan-type', default='l2', choices=['cs', 'l1', 'l2'], help='quantization types')
    parser.add_argument('--multiclass-loss', default='label_smoothing',
                        choices=['bce', 'imbalance', 'label_smoothing'], help='multiclass loss types')
    args = parser.parse_args()
    if "imagenet" in config['dataset']:
        multiclass = False
    elif "coco" in config['dataset']:
        multiclass = True
    elif "nus" in config['dataset']:
        multiclass = True
    else:
        print("check dataset if multi class,then add code at line 65 in main.py")
        exit(0)
    config['loss_param'] = {
    # loss_param
    'ce': args.ce,
    's': args.scale,
    'm': args.margin,
    "multiclass":multiclass,
    'm_type': args.margin_type,
    'quan': args.quan,
    'quan_type': args.quan_type,
    'multiclass_loss': args.multiclass_loss,
    'device': config['device']
    }
    config = config_dataset(config)
    return config


class HashCenter(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashCenter, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


def load_checkpoint(checkpoint_file, model):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()
    
    ## for imagenet pretrain first load , open below line
    #state_dict = {'backbone.'+k : v for k, v in state_dict.items()}
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        logger.info('All params loaded')
    else:
        logger.info('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        logger.info('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        logger.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    return ms

def infer(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    logger.info("train_loader: {} test_loader: {} dataset_loader: {}".format(len(train_loader), len(test_loader), len(dataset_loader)))
    config["num_train"] = num_train
    criterion = HashCenter(config, bit)
    codebook = criterion.hash_targets
    codebook = codebook.sign().to(device)
    net = config["net"](bit, config, codebook).to(device)
    ckpt_path = "checkpoints/model.pt"
    net = load_checkpoint(ckpt_path, net)
    net.eval()
    Best_mAP = 0
    print("calculating test binary code......")
    tst_data,tst_label = compute_result2(test_loader, net, device=device)
    print("calculating dataset binary code.......")
    trn_data, trn_label = compute_result2(dataset_loader, net, device=device)
    from numpy import linspace
    for thred in linspace(0.0001,0.01,50):

        logger.info("thred:%f, %s bit:%d, dataset:%s, training...." % (thred,
                config["info"], bit, config["dataset"]))
 
        tst_binary = process_thred(tst_data, thred)
        trn_binary = process_thred(trn_data, thred)
        print("calculating map.......")
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                            config["topK"]) 
        logger.info("thred:{}\t{}".format(thred, mAP))
        print("thred:{}\t{}".format(thred, mAP))
    tf_writer.add_scalar('mAP/infer', mAP, thred)
    epoch = 1
    if mAP > Best_mAP:
        Best_mAP = mAP
        tf_writer.add_scalar('best-mAP/infer', Best_mAP, epoch)
        if "save_path" in config:
            if not os.path.exists(config["save_path"]):
                os.makedirs(config["save_path"])
            np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary_infer.npy"),
                    trn_binary.numpy())
            logger.info("save trn_binary_infer to : {}".format(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary_infer.npy")))

    logger.info("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
        config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))

if __name__ == "__main__":
    config = get_config()
    print(config)

    for bit in config["bit_list"]:
        config['store_name'] = "infer"
        logger = setup_logger(output=os.path.join(config['root_log'], config['store_name']),
                            distributed_rank=0,
                            name=config['info'])
        tf_writer = SummaryWriter(log_dir=os.path.join(config['root_log'],config['store_name']))
        logger.info('logger storing name: ' + config['store_name'])
        config['save_path'] = os.path.join('checkpoints', config['store_name'])
        infer(config, bit)
