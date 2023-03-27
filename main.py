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
    
    parser = argparse.ArgumentParser(description='OrthoHash')

    # loss related params
    parser.add_argument('--scale', default=10, type=float, help='scale for cossim')
    parser.add_argument('--margin', default=0.15, type=float, help='ortho margin ')
    parser.add_argument('--margin-type', default='arc', choices=['cos', 'arc'], help='margin type')
    parser.add_argument('--ce', default=1.0, type=float, help='classification scale')
    parser.add_argument('--quan', default=1.0, type=float, help='quantization loss scale')
    parser.add_argument('--quan-type', default='cs', choices=['cs', 'l1', 'l2'], help='quantization types')
    parser.add_argument('--multiclass-loss', default='label_smoothing',
                        choices=['bce', 'imbalance', 'label_smoothing'], help='multiclass loss types')
    parser.add_argument('--signt', default=0.01, type=float, help='sign function thred')
    
    args = parser.parse_args()
    
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[network_patch_arc]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 50,
        "net": ResNet_DOLG,
        'S3_DIM':1024,
        "S4_DIM": 2048,
        'learn_cent':False,
        "dataset": "imagenet",
        "epoch": 500,
        "test_map": 5,
        "device": torch.device("cuda:0"),
        "bit_list": [64],
        'root_log': 'logs',
        'signt': args.signt,
    }

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


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    logger.info("train_loader: {} test_loader: {} dataset_loader: {}".format(len(train_loader), len(test_loader), len(dataset_loader)))
    config["num_train"] = num_train
    criterion = HashCenter(config, bit)
    codebook = criterion.hash_targets
    codebook = codebook.sign().to(device)
    net = config["net"](bit, config, codebook).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    Best_mAP = 0
    loss_param = config['loss_param']
    loss_func = OrthoHashLoss(**loss_param)
    logger.info("%s bit:%d, dataset:%s, training...." % (
            config["info"], bit, config["dataset"]))

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        net.train()
        train_loss = 0
        step = 0
        for image, label, ind in tqdm(train_loader):
            step += 1
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits, codes, norm_fea, thr = net(image, label.float())
            loss = loss_func(logits, codes, label, thr=thr)
            # if step % 100 == 0:
            #     logger.info("loss_ce:{}\loss_qua:{}".format(loss_ce, loss_qua ))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        logger.info("%s [%2d/%2d] loss:%.3f" % (config['info'], epoch+1, config['epoch'], train_loss))

        tf_writer.add_scalar('loss/train', train_loss, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
        if (epoch + 1) % config["test_map"]==0:
            with torch.no_grad():
                print("calculating test binary code......")
                tst_binary, tst_label, thr = compute_result2(test_loader, net, device=device)
                tst_binary = process_thred(tst_binary, thr)

                print("calculating dataset binary code.......")
                trn_binary, trn_label, thr = compute_result2(dataset_loader, net, device=device)
                trn_binary = process_thred(trn_binary, thr)

                print("calculating map.......")
                mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
                tf_writer.add_scalar('mAP/train', mAP, epoch)

            if mAP > Best_mAP:
                Best_mAP = mAP
                tf_writer.add_scalar('best-mAP/train', Best_mAP, epoch)
                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    save_name = os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-model.pt")
                    torch.save(net.state_dict(),save_name)
                    logger.info("save model to : {}".format(save_name))

            logger.info("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
def dump_config(config):
    dump_name = os.path.join(config['root_log'],config['store_name'], 'args.txt')
    lines = []
    for k,v in config.items():
        if k == 'net':
            line = "{}  :  {} \n".format(k,v.__name__)
        else:
            line = "{}  :  {} \n".format(k,str(v))
        lines.append(line) 
    with open(dump_name, 'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    config = get_config()
    print(config)

    for bit in config["bit_list"]:
        config['store_name'] = "{}_{}_{}_epoch_{}_bit_{}_bs_{}_crop_{}_scale_{}_margin_{}_{}_{}".format(config['info'], config['net'].__name__, config['dataset'], str(config['epoch']), str(bit), str(config['batch_size']), str(config['crop_size']), str(config['loss_param']['s']), str(config['loss_param']['m']), str(config['loss_param']['m_type']), str(config['signt']))
        logger = setup_logger(output=os.path.join(config['root_log'], config['store_name']),
                            distributed_rank=0,
                            name=config['info'])
        tf_writer = SummaryWriter(log_dir=os.path.join(config['root_log'],config['store_name']))
        logger.info('logger storing name: ' + config['store_name'])
        config['save_path'] = os.path.join('checkpoints', config['store_name'])
        dump_config(config)
        train_val(config, bit)
