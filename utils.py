import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml
import numpy as np
import torch
import pandas as pd
from train_utils import ce_loss
import torch.nn as nn
class Teacher:
    """
    This class enables teacher training without requiring model inference via a lookup table
    """
    def __init__(self, num_samples = 50000,num_classes = 10,teacher_threshold = 0.99,device = 0):
        self.threshold = teacher_threshold
        self.num_classes = num_classes
        self.device = device
        self.lossfn = nn.CrossEntropyLoss()
        self.probabilites_by_idx = torch.zeros((num_samples,num_classes ), dtype=torch.float ).cuda(device = device)
        self.prediction_count =  torch.zeros((num_samples,), dtype=torch.long ).cuda(device = device)
        self.averaged_teacher_logits = torch.zeros((num_samples,num_classes ), dtype=torch.float ).cuda(device = device)

    def set_device(self, device):
        self.device = device
        self.probabilities_by_index = self.probabilites_by_idx.cuda(device = self.device)
        self.prediction_count.cuda(device = self.device)
        self.averaged_teacher_logits.cuda(device = self.device) 
    def update(self, logits, x_index):
        with torch.no_grad():
            logits = logits.cuda(device = self.device)
            prob = torch.softmax(logits, dim=1)
            prob_value , class_value = torch.max(prob, dim = 1)
            mask = prob_value.ge(self.threshold).long()
            masked_probs = mask * prob.transpose(0,1)
            self.probabilites_by_idx[x_index].cuda(device = self.device) = masked_probs.transpose(0,1).cuda(device = self.device)
            self.prediction_count[x_index] += mask
    def clean(self,arr):
        arr[arr!=arr] = 0  
        arr[arr == float("inf")] = 0
        return arr


        
    def graduate_teacher(self):

        self.averaged_teacher_logits += (self.probabilites_by_idx.transpose(0,1) / self.prediction_count).transpose(0,1)
        self.averaged_teacher_logits = self.clean(self.averaged_teacher_logits)
        self.active = (self.prediction_count != 0)
        print("Total number of active labels : {}".format(torch.sum(self.active)))
        self.probabilites_by_idx = torch.zeros((num_samples,num_classes ), dtype=torch.float ).cuda(device = device)
        self.prediction_count =  torch.zeros((self.num_samples,), dtype=torch.long ).cuda(device = self.device)

    def loss(self, student_logits,x_index):
        samples_active = self.active[x_index]
        student_logits_active = student_logits[samples_active,:]
        teacher_logits_active = self.averaged_teacher_logits[x_index][samples_active,:]
        return self.lossfn(student_logits_active,teacher_logits_active)


class Label_Metrics:
    def __init__(self, train):
        self.labels = torch.from_numpy(np.asarray(train.get_correct_labels())).cuda()
        self.acc_df = pd.DataFrame(columns = ["u_acc", "iteration", "true accuracy", "threshold","quantity"])
    def eval_batch(self, pseudolabels,index): 
        correct = np.sum(np.where(self.labels[index.cpu().numpy()] == pseudolabels.cpu().numpy(),1,0))
        active = len(pseudolabels)
        print("Correct labels in batch = {}".format(correct))
        print("percentage accuracy in batch : {}".format(correct/active))
        print("Total number of labels in batch = {}".format(active))

    def eval_total(self,selected_label):
         
        correct_labels = self.labels
        correct = torch.sum(self.labels == selected_label).item()
        active = torch.sum(selected_label != -1).item()
        print("Total number of correct labels = {}".format(correct))
        print("percentage accuracy : {}".format(correct/active if active != 0 else 0))
        print("Total number of labels active = {}".format(active))

    def prob_eval(self,logits, x_index,acc,it):
        print("logits shape: {}".format(logits.shape))
        prob = torch.softmax(logits, dim=1)
        prob_value , class_value = torch.max(prob, dim = 1)
        correct = (self.labels[x_index] == class_value)
        num_samples = len(x_index)
        for threshold in range(90,100):
            mask = prob_value.ge(threshold*0.01)
            out = correct * mask
            self.acc_df.loc[len(self.acc_df)] = [(torch.sum(out)/torch.sum(mask)).cpu().item() if torch.sum(mask) != 0 else 0, it , acc , threshold, torch.sum(mask).cpu().item()]
            if torch.any(mask):
                print("Accuracy at threshold ({}) with {} labels: {}".format(threshold ,torch.sum(mask),torch.sum(out)/torch.sum(mask)))
        self.acc_df.to_csv("teacher.csv", index=False)


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'

    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c': 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]

    else:
        if net_name == 'WideResNet':
            import models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'WideResNetVar':
            import models.nets.wrn_var as net
            builder = getattr(net, 'build_WideResNetVar')()
        elif net_name == 'ResNet50':
            import models.nets.resnet50 as net
            builder = getattr(net, 'build_ResNet50')(is_remix)
        else:
            assert Exception("Not Implemented Error")

        if net_name != 'ResNet50':
            setattr_cls_from_kwargs(builder, net_conf)
        return builder.build


def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
