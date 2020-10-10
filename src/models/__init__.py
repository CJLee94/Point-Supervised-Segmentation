import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from . import base_networks
# from haven import haven_results as hr 
from haven import haven_chk as hc
from . import lcfcn
from . import PENet
from . import PONet


def get_model(model_dict, exp_dict=None, train_set=None):
    if model_dict["name"] in ["lcfcn"]:
        model =  lcfcn.LCFCN(exp_dict, train_set=train_set)
    elif model_dict["name"] in ["penet"]:
        model = PENet.PENet(exp_dict, train_set =train_set)
    elif model_dict["name"] in ["ponet"]:
        model = PONet.PONet(exp_dict['model'], train_set =train_set)
    return model
        