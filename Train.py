# import glob
import os
from trainval import trainval
from haven import haven_utils as hu
import shutil
import argparse
# import matplotlib.pyplot as plt
# %matplotlib inline

# Define the dataset you want to train on, you can type 'TNBC' or 'MoNuSegTrainingData'

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_dict', required=True, type=str)

args = parser.parse_args()

datadir = './CoNSeP/' 

# exp_dict_file = 'exp_config_ponet.json'
# exp_dict_file = args.exp_dict

exp_dict = hu.load_json(args.exp_dict)
savedir_base = "./Result"
os.makedirs(savedir_base, exist_ok=True)
trainval(exp_dict,
         savedir_base,
         datadir,
         reset=False,
         num_workers=25
         )
shutil.copy(args.exp_dict, os.path.join(savedir_base, hu.hash_dict(exp_dict), args.exp_dict))
