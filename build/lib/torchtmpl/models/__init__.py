# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
# from .foundation_models import *
from .resnet_models import *


def build_model(cfg, input_size, num_classes):
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
