import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.core.fromnumeric import argsort, mean, squeeze
from torch import tensor
from torch.functional import norm
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


if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

Path1=r"C:\MMaster\Files"

#from google.colab import drive

#drive.mount('/content/drive') 

class NLR3(nn.Module):
    def __init__(self,netin,netout,nethidden):
      super().__init__()
      self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden),torch.nn.Tanh(),torch.nn.Linear(nethidden, netout))
    def myforward (self,inv):
      outv=self.netmodel(inv)
      return outv


with open(Path1+'\dataset172Org\Features172KphixTarget.txt', 'rb') as fp:
       target=pickle.load( fp)

with open(Path1+'\dataset172Org\Features172KphitTargetCaption.txt', 'rb') as fp:
       inp=pickle.load( fp)

inp=torch.tensor(inp)
target=torch.tensor(target)
cnt=np.zeros(5)
rng=[1,5,10,50,100]
rng=np.array(rng)
hidden=1000
l_r=0.1
epoch=25000
tag='finalstagexttx'
batch_size=1000
save_duration=500
seed=100
min_error=10
model_mlp=NLR3(inp.shape[1],target.shape[1],hidden).to(device)
model_mlp.load_state_dict(torch.load( Path1+'/4netcpttrgt.pth', map_location=torch.device('cpu') ))
loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model_mlp.parameters(), lr=l_r)
s=0
sweep_range=inp.shape[0]%batch_size

totallosses=[]

for j in range(epoch):
  total_loss=0
  
  for l in range(int(inp.shape[0]/batch_size)):
    
    item_batch = inp[l*batch_size+s:(l+1)*batch_size+s,:].to(device)

    target_batch=target[l*batch_size+s:(l+1)*batch_size+s,:].to(device)
    netoutbatch=model_mlp.myforward(item_batch)
    loss = loss_fn(target_batch,netoutbatch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss+=loss
  if (total_loss<min_error):
    break
  #print('iteration:',j,' loss ',loss, 'total loss',total_loss)
  print('iteration:',j, 'total loss',total_loss,'avg loss', total_loss/(inp.shape[0]/batch_size))
  totallosses.append(total_loss)
  s+=1
  if s==sweep_range:
      s=0
  if (j%save_duration==0) :
   torch.save(model_mlp.state_dict(), Path1+'\4netcpttrgt.pth') 

    #torch.save(model_mlp.state_dict(), Path1+r'\4final_net_with_Shup'+str(j)+r'.pth') 
   with open(Path1+'\4lossesnetcpttrgt.pkl', 'wb') as fp:
       pickle.dump( totallosses, fp)


print('Finished Training')
torch.save(model_mlp.state_dict(), Path1+'\4final_net4cpttrgt.pth') 
