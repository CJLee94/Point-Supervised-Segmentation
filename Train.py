# import glob
import os
from trainval import trainval
from haven import haven_utils as hu
# import matplotlib.pyplot as plt
# %matplotlib inline

# Define the dataset you want to train on, you can type 'TNBC' or 'MoNuSegTrainingData'
datadir = './CoNSeP/' 

exp_dict = hu.load_json('exp_config_ponet.json')
savedir_base = "./Result"
os.makedirs(savedir_base, exist_ok=True)
trainval(exp_dict,
         savedir_base,
         datadir,
         reset=True,
         num_workers=25
         )
