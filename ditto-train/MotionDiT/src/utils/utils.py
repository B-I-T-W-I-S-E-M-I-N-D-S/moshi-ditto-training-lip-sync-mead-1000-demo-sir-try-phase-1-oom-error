import json
import random
import os
import numpy as np
import torch
import time
import uuid
import pickle


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, pkl):
    with open(pkl, "wb") as fw:
        pickle.dump(obj, fw)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def random_a_seed(num=1e6):
    random.seed(None)
    seed = int(random.random() * num)
    return seed


def random_tmp():
    tmp = "{}_{}".format(int(time.time() * 1000), str(uuid.uuid4()).replace("-", ""))
    return tmp


class DictAverageMeter:
    def __init__(self):
        self.keys = None
        self.sum = None
        self.count = None
        self.avg = None
        self.val = None
        self.initialized = False

    def init(self, val):
        self.keys = set(val.keys())
        self.val = val
        self.sum = val
        self.avg = val
        self.count = {k: 1 for k in val}
        self.initialized = True

    def add(self, val):
        self.keys = self.keys | set(val.keys())
        self.val = val
        self.sum = {k: self.sum.get(k, 0) + val.get(k, 0) for k in self.keys}
        self.count = {
            k: self.count.get(k, 0) + (1 if k in val else 0) for k in self.keys
        }
        self.avg = {k: self.sum[k] / self.count[k] for k in self.keys}

    def update(self, val):
        if not self.initialized:
            self.init(val)
        else:
            self.add(val)

    def value(self):
        return self.val

    def average(self):
        return self.avg
    