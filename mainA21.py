from numpy.core.fromnumeric import mean
import torch
from torch.functional import norm
import torchvision
import torchvision.transforms as tvt
import torch.nn as nn
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from torch import optim
import torch.nn.functional as F
import math as m
import time
import os 
#from google.colab import drive
import random
from PIL import Image
from torch.autograd import Variable, variable
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
#import datasets
from tqdm import tqdm as tqdm
import PIL
import argparse
import datasets
import img_text_composition_models


#Path1=r"C:\MMaster\Files"

Path1=r"D:\personal\master\MyCode\files"
#################  Support Functions Section   #################


def build_and_train_netCOS(hidden1,hidden2,max_iterations, min_error, all_queries,all_imgs,batch_size):
  all_queries=Variable(torch.Tensor(all_queries))
  all_imgs=variable(torch.tensor(all_imgs))
  model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)
  #model=model.cuda()
  torch.manual_seed(3)
  loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  torch.manual_seed(3)
  #criterion = nn.CosineSimilarity() 
  criterion=nn.CosineSimilarity(dim=1, eps=1e-6)
 #loss.backward()

  optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
  epoch=max_iterations

  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    for l in range(int(all_queries.shape[0]/batch_size)):
      
      item_batch = all_queries[l*batch_size:(l+1)*batch_size-1,:]
      target_batch=all_imgs[l*batch_size:(l+1)*batch_size-1,:]
      netoutbatch=model.myforward(item_batch)
      #loss = loss_fn(target_batch,netoutbatch)
      loss = torch.mean(torch.abs(1-loss_fn(target_batch,netoutbatch)))

      #loss=1-loss
      losses.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
      if (l%1000==0) :
        print('Epoch:',j,' get images batch=',l*batch_size,':',(l+1)*batch_size,'loss',loss,end='\r')
    if (total_loss<min_error):
      break
    print('iteration:',j, 'total loss',total_loss)
    totallosses.append(total_loss)
    if(j%1000==0):
      torch.save(model.state_dict(), Path1+r'\NLPCOS172K'+str(j)+'.pth') 

  print ('mean square loss',loss_fn(model.myforward(all_queries),all_imgs))  
  print('Finished Training')
  with open(Path1+r"/"+'loosses2.pkl', 'wb') as fp:
          pickle.dump( totallosses, fp)

  torch.save(model.state_dict(), Path1+r'\NLPCOSfinal172k.pth') 

class NLR2(nn.Module):
  def __init__(self,netin,netout,nethidden1,nethidden2):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.Sigmoid(),torch.nn.Linear(nethidden1, nethidden2),torch.nn.Linear(nethidden2, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

def ab_Mgetvaluesfilesaved():

  trainset = datasets.Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))
  
  testset = datasets.Fashion200k(
        path=Path1,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in trainset.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'
  

  for x in range(1000, 3001, 1000):
    asbook1, model, euc_model = ab_MtestLoaded(opt, trig, testset,x)
    print(x,': ',asbook1, '\n  model generated      ',model, '\n    euc model',euc_model )

  # for name, dataset in [ ('train', trainset)]:
  #   R = test_retrieval.testLoaded(opt, trig, dataset)
  #   print(name,' PAPER Results: ',R)

def ab_MtestLoaded(opt, model, testset,option):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()
  
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    
    all_imgs = datasets.Features33K().Get_all_images()
    all_captions = datasets.Features33K().Get_all_captions()
    all_queries = datasets.Features33K().Get_all_queries()
    all_target_captions = datasets.Features33K().Get_target_captions()
    new_all_queries=mymodels(all_queries,all_imgs,option)

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images()[:10000]
    
    all_captions = datasets.Features172K().Get_all_captions()[:10000]
    all_queries = datasets.Features172K().Get_all_queries()[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions()[:10000]
    
    new_all_queries=mymodels(all_queries,all_imgs,option)

  # feature normalization
  diff=new_all_queries-all_queries
  diff=diff**2
  diff=np.sum(diff,axis=1)
  print(mean(diff))

  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    new_all_queries[i, :] /= np.linalg.norm(new_all_queries[i, :])

  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  new_nn_result = []
  euc_new_nn_result=[]
  for i in tqdm(range(all_queries.shape[0])):
    #sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    new_sims = new_all_queries[i:(i+1), :].dot(all_imgs.T)
    #euc_new_sims=np.sum(abs(all_imgs-all_queries[i, :]),axis=1)
    if test_queries:
    #  sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      new_sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
    #  euc_new_sims[test_queries[i]['source_img_id']]=10e10

    #nn_result.append(np.argsort(-sims[0, :])[:110])
    new_nn_result.append(np.argsort(-new_sims[0, :])[:110])
    #euc_new_nn_result.append(np.argsort(euc_new_sims)[:110])

  # compute recalls
  out = []
  out2=[]
  out3=[]
  #nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  new_nn_result = [[all_captions[nn] for nn in nns] for nns in new_nn_result]
  #euc_new_nn_result = [[all_captions[nn] for nn in nns] for nns in euc_new_nn_result]

  for k in [1, 5, 10, 50, 100]:
    # r = 0.0
    # for i, nns in enumerate(euc_new_nn_result):
    #   if all_target_captions[i] in nns[:k]:
    #     r += 1
    # r /= len(euc_new_nn_result)
    # #out += [('recall_top' + str(k) + '_correct_composition', r)]
    # out3.append(str(k) + ' ---> '+ str(r*100))
    
    r = 0.0
    for i, nns in enumerate(new_nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(new_nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out2.append(str(k) + ' ---> '+ str(r*100))

    # r = 0.0
    # for i, nns in enumerate(nn_result):
    #   if all_target_captions[i] in nns[:k]:
    #     r += 1
    # r /= len(nn_result)
    # #out += [('recall_top' + str(k) + '_correct_composition', r)]
    # out.append(str(k) + ' ---> '+ str(r*100))

    
  return out, out2, out3

def mymodels(all_queries,all_imgs,option):
    hidden1=950
    hidden2=850
    batch_size=200
    itr=15000
    
    model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)

    model.load_state_dict(torch.load(Path1+r'\NLP2COS172K'+str(option)+'.pth'))
    model.eval()
    all_queries=Variable(torch.Tensor(all_queries))
 
    new_all_queries=model.myforward(all_queries)
    new_all_queries = torch.tensor(new_all_queries,requires_grad=False)
    new_all_queries=np.array(new_all_queries)
    
    return new_all_queries


  
if __name__ == '__main__': 
  ab_Mgetvaluesfilesaved()
