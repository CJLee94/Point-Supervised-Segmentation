import glob,os
from trainval import trainval
from haven import haven_utils as hu
import matplotlib.pyplot as plt
# %matplotlib inline

# Define the dataset you want to train on, you can type 'TNBC' or 'MoNuSegTrainingData'
datadir = './CoNSeP/' 

# the exp_dict is the configuration of the model which has already been saved. 
# It consists of the initial learning rate, max epoch, model name and optimizer
# The saved one looks like this
# {
#     "batch_size": 1,
#     "lr": 0.001,
#     "max_epoch": 20,
#     "model": {
#         "name": "penet"
#     },
#     "optimizer": "adam"
# }
exp_dict = hu.load_json('exp_config_ponet.json')
exp_dict["batch_size"] = 64
exp_dict["max_epoch"] = 100
# folddir_10 = glob.glob(os.path.join(datadir, '*_fold'))

# for folddir in folddir_10:
#     savedir_base = os.path.join(folddir,'Result')
#     os.makedirs(savedir_base,exist_ok=True)
savedir_base = "./Result"
os.makedirs(savedir_base,exist_ok=True)
trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=25)