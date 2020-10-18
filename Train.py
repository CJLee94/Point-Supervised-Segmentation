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
parser.add_argument('-d', '--data_dir', type=str, default='./CoNSeP/')
parser.add_argument('-s', '--save_dir', type=str, default='./Result')

args = parser.parse_args()

datadir = args.data_dir

exp_dict = hu.load_json(args.exp_dict)
savedir_base = args.save_dir
os.makedirs(savedir_base, exist_ok=True)
trainval(exp_dict,
         savedir_base,
         datadir,
         reset=False,
         num_workers=12
         )
shutil.copy(args.exp_dict,
            os.path.join(savedir_base,
                         hu.hash_dict(exp_dict),
                         os.path.split(args.exp_dict)[-1]))
