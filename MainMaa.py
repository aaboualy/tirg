from numpy.core.fromnumeric import argsort, squeeze
from tensorflow.python.ops.array_ops import zeros
from tensorflow.python.ops.gen_array_ops import concat
import torch
from torch import tensor
from torch.functional import norm
#from torch._C import float32
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




Path1=r"C:\MMaster\Files\phase2"
path2=r"C:\MMaster\Files"


def PrintSizeDatasets():
    ########## 172K #########

    print('Querys Imgs Lenght 172k:',len(datasets.Feature172KOrg().PhixQueryImg))
    print('Querys Captions Lenght 172k:',len(datasets.Feature172KOrg().PhitQueryCaption))
    print('Querys Modifier Text Lenght 172k:',len(datasets.Feature172KOrg().PhitQueryMod))
    print('Target Imgs Lenght 172k:',len(datasets.Feature172KOrg().PhixTargetImg))
    print('Target Captions Lenght 172k:',len(datasets.Feature172KOrg().PhitTargetCaption))
 
    ########## 33K #########

    print('Querys Imgs Lenght 33K:',len(datasets.Features33KOrg().PhixQueryImg))
    print('Querys Captions Lenght 33K:',len(datasets.Features33KOrg().PhitQueryCaption))
    print('Querys Modifier Text Lenght 33K:',len(datasets.Features33KOrg().PhitQueryMod))
    print('Target Imgs Lenght 33K:',len(datasets.Features33KOrg().PhixTargetImg))
    print('Target Captions Lenght 33K:',len(datasets.Features33KOrg().PhitTargetCaption))
    print('All Imgs Unique Lenght 33K:',len(datasets.Features33KOrg().PhixAllImages))
    print('All Imgs Captions Unique Lenght 33K:',len(datasets.Features33KOrg().PhitAllImagesCaptions))




if __name__ == '__main__': 
    PrintSizeDatasets()
    
    
  