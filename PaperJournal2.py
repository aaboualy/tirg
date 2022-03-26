from numpy.core.fromnumeric import argsort, squeeze
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.gen_array_ops import concat
import torch
from torch import tensor
from torch.functional import norm
# from torch._C import float32
import torchvision
import torchvision.transforms as tvt
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn.functional as F
import math as m
import time
import os
import random
from PIL import Image
from torch.autograd import Variable
from PIL import Image
import numpy
import tensorflow as tf
from pathlib import Path
import pickle
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import test_retrieval
import torch_functions
from tqdm import tqdm as tqdm
import PIL
import argparse
import datasets
import img_text_composition_models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


Path1 = r"C:\MMaster\Files"
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")


class NLR3S(nn.Module):
    def __init__(self,netin,netout,nethidden):
      super().__init__()
      self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden),torch.nn.Sigmoid(),torch.nn.Linear(nethidden, netout))
    def myforward (self,inv):
      outv=self.netmodel(inv)
      return outv
class NLR3T(nn.Module):
    def __init__(self,netin,netout,nethidden):
      super().__init__()
      self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden),torch.nn.Tanh(),torch.nn.Linear(nethidden, netout))
    def myforward (self,inv):
      outv=self.netmodel(inv)
      return outv


def Semantic50_Maa(run_type):
    device = torch.device("cpu")    

    if run_type=='train': 
        PhixQueryImg =datasets.Features152Org().phix_152[:10000]
        PhitQueryCaption =datasets.Feature172KOrg().PhitQueryCaption[:10000]
        PhitQueryMod =datasets.Feature172KOrg().PhitQueryMod[:10000]
        PhixTargetImg =datasets.Features152Org().target_phix_152[:10000]
        PhitTargetCaption =datasets.Feature172KOrg().PhitTargetCaption[:10000]
        all_captions_text =datasets.Feature172KOrg().all_captions_text[:10000]
        all_target_captions_text =datasets.Feature172KOrg().all_target_captions_text[:10000]
        all_Query_captions_text =datasets.Feature172KOrg().all_Query_captions_text[:10000]
        all_ids =datasets.Feature172KOrg().all_ids[:10000]
        SearchedFeatures=PhixTargetImg
        

    elif run_type=='test':    
        PhixQueryImg=datasets.Features152Org().phix_152_test
        PhitQueryCaption=datasets.Features33KOrg().PhitQueryCaption
        PhitQueryMod=datasets.Features33KOrg().PhitQueryMod
        PhixTargetImg=datasets.Features152Org().target_phix_152_test
        PhitTargetCaption=datasets.Features33KOrg().PhitTargetCaption
        PhixAllImages=datasets.Features33KOrg().PhixAllImages
        PhitAllImagesCaptions=datasets.Features33KOrg().PhitAllImagesCaptions
        all_captions_text=datasets.Features33KOrg().all_captions_text
        all_target_captions_text=datasets.Features33KOrg().all_target_captions_text
        all_Query_captions_text=datasets.Features33KOrg().all_queries_captions_text
        all_queries_Mod_text=datasets.Features33KOrg().all_queries_Mod_text
        all_ids=datasets.Features33KOrg().all_ids
        SearchedFeatures=PhixAllImages

    
    PhitQueryCaption=torch.tensor(PhitQueryCaption).to(device)
    PhixQueryImg=torch.tensor(PhixQueryImg).to(device)
    PhitQueryMod=torch.tensor(PhitQueryMod).to(device)
    PhitTargetCaption=torch.tensor(PhitTargetCaption).to(device)
    PhixTargetImg=torch.tensor(PhixTargetImg).to(device)

    phix=PhixQueryImg
    phit=PhitQueryMod
    


    hidden=1000
    NetA=NLR3T(phix.shape[1],phit.shape[1],hidden)
    NetA.load_state_dict(torch.load( Path1+r'/NetANN50UpTR.pth', map_location=torch.device('cpu') ))
    hidden=2500
    NetB=NLR3T(phit.shape[1],phix.shape[1],hidden)
    NetB.load_state_dict(torch.load( Path1+r'/NetB50_2500Upt1803.pth', map_location=torch.device('cpu') ))    
    hidden=1800
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],hidden)
    NetC.load_state_dict(torch.load( Path1+r'/NetCfinalUpTR.pth', map_location=torch.device('cpu') ))


    
    NetAout=NetA.myforward(phix)    
    NetCinp=torch.cat((phit,NetAout[:phit.shape[0],:]),1)
    NetCout=NetC.myforward(NetCinp)    
    net_target=NetB.myforward(NetCout)


    
    nn_result = []
    #phixN=torch.tensor(phixN)
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])
    net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])

    for i in range (phix.shape[0]):  #(3900): #
        sims = net_target[i, :].dot(phix[:net_target.shape[0],:].T)
        #print(i)
        nn_result.append(np.argsort(-sims[ :])[:110])
    
        
    # compute recalls
    out = []
    nn_result = [[all_Query_captions_text[nn] for nn in nns] for nns in nn_result]
    
    
    for k in [1, 5, 10, 50, 100]:
        
        r = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions_text[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
        #out += [('recall_top' + str(k) + '_correct_composition', r)]
        out.append(str(k) + ' ---> '+ str(r*100))
        r = 0.0

    print (out)




if __name__ == '__main__': 
    Semantic50_Maa("train")
    Semantic50_Maa("test")
    