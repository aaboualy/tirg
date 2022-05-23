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
import torchvision.models as models


Path1 = r"C:\MMaster\Files"
print(torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")

else:
    device = torch.device("cpu")
    print("CPU")


def euclideandistance(signature,signatureimg):
    from scipy.spatial import distance
    return distance.euclidean(signature, signatureimg)



#Paper 1
########################################################################################################################################
#linear regression

#by Train Dataset

def getbetatrain():
  
  
  train = datasets.Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in train.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()

  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]
 
  
  for Data in tqdm(train):
    
    
    imgs += [Data['source_img_data']]
    mods += [Data['mod']['str']]
    target +=[Data['target_img_data']]
    
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    f2 = trig.extract_img_feature(target).data.cpu().numpy()

    trigdata.append(f[0])
    imgdata.append(f2[0])
    
    imgs = []
    mods = []
    target = []

    
  
  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)

  Ntrigdata=trigdata
  Nimgdata=imgdata

  Ntrig2=[]
  for i in range(Ntrigdata.shape[0]):
    Ntrigdata[i, :] /= np.linalg.norm(Ntrigdata[i, :])
  for i in range(Nimgdata.shape[0]):
    Nimgdata[i, :] /= np.linalg.norm(Nimgdata[i, :])
  for i in range(Ntrigdata.shape[0]):
    Ntrig2.append(np.insert(Ntrigdata[i],0, 1))


  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,Nimgdata) 

  

  with open(Path1+r"/"+'BetaNot.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestrain():
  
  with open (Path1+"\\BetaNot.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 

  

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
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
    
    betaNor = test_retrieval.testWbeta(opt, trig, dataset,BetaNormalize)
    print(name,' Beta tain Normalized: ',betaNor)

#by test dataset

def getbetatest():
  train = datasets.Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  test = datasets.Fashion200k(
        path=Path1,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in train.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()


  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]
 
  all_source_captions=[]
  all_target_captions=[]

  
  for Data in tqdm(test.get_test_queries()):
    imgs += [test.get_img(Data['source_img_id'])]
    mods += [Data['mod']['str']]
    target +=[test.get_img(Data['target_id'])]
    all_source_captions +=Data['source_caption']
    all_target_captions +=Data['target_caption']

    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    f2 = trig.extract_img_feature(target).data.cpu().numpy()

    
    trigdata.append(f[0])
    imgdata.append(f2[0])

    
    imgs = []
    mods = []
    target = []

 

  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)
  
  Ntrigdata=trigdata
  Nimgdata=imgdata

  Ntrig2=[]
  trigdata2=[]

  for i in range(Ntrigdata.shape[0]):
    Ntrigdata[i, :] /= np.linalg.norm(Ntrigdata[i, :])
  
  for i in range(Nimgdata.shape[0]):
    Nimgdata[i, :] /= np.linalg.norm(Nimgdata[i, :])


  for i in range(Ntrigdata.shape[0]):
      Ntrig2.append(np.insert(Ntrigdata[i],0, 1))

  

  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,Nimgdata) 

  


  with open(Path1+r"/"+'Betatest.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestest():
  
  with open (Path1+"\\Betatest.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 

  

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
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
    
    betaNor = test_retrieval.testWbeta(opt, trig, dataset,BetaNormalize)
    print(name,' Beta test Normalized: ',betaNor)

# by joint datset

def GetAverageBeta():
  with open (Path1+"/BetaNot.txt", 'rb') as fp:
    BetaTrain = pickle.load(fp) 

  with open (Path1+"/Betatest.txt", 'rb') as fp:
    BetaTest = pickle.load(fp) 

  BetaAvg1= np.add( BetaTest,BetaTrain)
  BetaAvg2=BetaAvg1/2

  
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
  
  for name, dataset in [ ('train', trainset),('test', testset)]: 
    
    betaNor = test_retrieval.testWbeta(opt, trig, dataset,BetaAvg2)
    print(name,' Beta Avg: ',betaNor)


# NLP

class NLR2(nn.Module):
  def __init__(self,netin,netout,nethidden1,nethidden2):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.Sigmoid(),torch.nn.Linear(nethidden1, nethidden2),
    torch.nn.Linear(nethidden2, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

def build_and_train_netMSE():


  hidden1=1050
  hidden2=950
  batch_size=500
  max_iterations=25000
  min_error=0.01
  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]

  
  train = datasets.Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in train.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()

  

  for Data in tqdm(train):
    
    
    imgs += [Data['source_img_data']]
    mods += [Data['mod']['str']]
    target +=[Data['target_img_data']]
    
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    f2 = trig.extract_img_feature(target).data.cpu().numpy()

    trigdata.append(f[0])
    imgdata.append(f2[0])
    
    imgs = []
    mods = []
    target = []

    
  
    #   all_queries=np.array(trigdata)
    #   all_imgs=np.array(imgdata) 


    #   all_queries=Variable(torch.Tensor(all_queries))
    #   all_imgs=Variable(torch.tensor(all_imgs))
  
  all_queries=trigdata
  all_imgs=imgdata


  model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)
 
  torch.manual_seed(3)
  loss_fn = torch.nn.MSELoss()
  torch.manual_seed(3)
  
  criterion=nn.MSELoss()
 

  optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
  epoch=max_iterations

  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    for l in range(int(50000/batch_size)):
      
      item_batch = all_queries[l*batch_size:(l+1)*batch_size-1,:]
      target_batch=all_imgs[l*batch_size:(l+1)*batch_size-1,:]
      netoutbatch=model.myforward(item_batch)
      loss = loss_fn(target_batch,netoutbatch)
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
    if (j%1000==0) :
      torch.save(model.state_dict(), Path1+r'\GNLPMSEt'+str(j)+r'.pth') 

  print ('mean square loss',loss_fn(model.myforward(all_queries),all_queries))  
  print('Finished Training')
  torch.save(model.state_dict(), Path1+r'\NLPMSEtFinal.pth') 

def resultsNLPMSE():
  
  
  hidden1=1050
  hidden2=950


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

  model=NLR2(512,512,hidden1,hidden2)
  model.load_state_dict(torch.load(Path1+r'\NLPMSEtFinal.pth' , map_location=torch.device('cpu') ))
  model.eval()
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
    NLP = test_retrieval.testLoadedNLP(opt, trig, dataset, model)
    print(name,'NLP Mean:',':-->',NLP)


# other approches

def GetValuesRegModel():

  
  train = datasets.Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in train.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()

  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]
 
  
  for Data in tqdm(train):
    
    
    imgs += [Data['source_img_data']]
    mods += [Data['mod']['str']]
    target +=[Data['target_img_data']]
    
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    f2 = trig.extract_img_feature(target).data.cpu().numpy()

    trigdata.append(f[0])
    imgdata.append(f2[0])
    
    imgs = []
    mods = []
    target = []

    
  
  all_queries1=np.array(trigdata)
  imgdata=np.array(imgdata)

  for i in range(all_queries1.shape[0]):
    all_queries1[i, :] /= np.linalg.norm(all_queries1[i, :])
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])

  reg = LinearRegression().fit(all_queries1, imgdata)

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
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
    # asbook = test_retrieval.test(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)
    
    betaNor = test_retrieval.testLoadedRegModel(opt, trig, dataset,reg)
    print(name,'Reg Model: ',betaNor)

def GetValuesRandomForestRegressor():
  SizeI=50000
  n_estimator=50
  max_depths=25

  train = datasets.Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in train.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()

  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]
 
  
  for Data in tqdm(train):
    
    
    imgs += [Data['source_img_data']]
    mods += [Data['mod']['str']]
    target +=[Data['target_img_data']]
    
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    f2 = trig.extract_img_feature(target).data.cpu().numpy()

    trigdata.append(f[0])
    imgdata.append(f2[0])
    
    imgs = []
    mods = []
    target = []


  imgdata = imgdata[:SizeI]
  all_queries1 = trigdata[:SizeI]

  for i in range(all_queries1.__len__()):
    all_queries1[i, :] /= np.linalg.norm(all_queries1[i, :])
  for i in range(imgdata.__len__()):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])

  print('1')
  regr = RandomForestRegressor(n_estimators=n_estimator,  random_state=0,max_depth=max_depths,verbose=True,bootstrap=False) #,min_samples_leaf=4,bootstrap=True,
  print('2')
  regr.fit(all_queries1, imgdata)
  print('3')


  filename = 'RFRN'+str(n_estimator)+'M'+str(max_depths)+'I'+str(SizeI)+'.pth'
  pickle.dump(regr, open(Path1+'/'+filename, 'wb'))
  regr = pickle.load(open(Path1+'/'+filename, 'rb'))
  

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
  
  for name, dataset in [ ('train', trainset)]: #('train', trainset),  ,('test', testset)
    
    betaNor = test_retrieval.testLoadedRandomForestRegressor(opt, trig, dataset,regr)
    print(name,' Random: ',betaNor)





#Paper2
########################################################################################################################################



# Due to need for high GPU Colab was used and for more efficiency All Features has been saved to files

def SaveFeaturesToFiles():
    datasets.FeaturesToFiles172().SaveAllFeatures()
    datasets.FeaturesToFiles33().SaveAllFeatures()


# Files in ColabFiles has been trained on Colab to genrate the below Models

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

def Semantic18_5(run_test):

    if run_test==0:
        with open (Path1+r'/FeaturesToFiles172/Features172QueryStructureallF.txt', 'rb') as fp:
            AllData = pickle.load(fp)         
        AllData=AllData[:10000]
    
    elif run_test==1:
        with open (Path1+r'/FeaturesToFiles33/Features33QueryStructureallF.txt', 'rb') as fp:
            AllData = pickle.load(fp) 

    

    
    phix=[d['Query18F'] for d in AllData]
    phit=[d['ModF'] for d in AllData]
    target=[d['Target18F'] for d in AllData]

    #phix,target,phit =getfeatures(run_test)


    phix=torch.tensor(phix)
    phit=torch.tensor(phit)
    phit=torch.squeeze(phit)
   
    target=torch.tensor(target)
    #allquerycaptions=[d['QueryCaption'] for d in AllData]
    

    NetA=NLR3T(phix.shape[1],phit.shape[1],1000)    
    NetA.load_state_dict(torch.load( Path1+r'/UltraNetA18tune.pth', map_location=torch.device('cpu') ))
   
    NetB=NLR3T(phit.shape[1],phix.shape[1],2500)
    NetB.load_state_dict(torch.load( Path1+r'/UltraNetB18_CO2804_best_19_final.pth', map_location=torch.device('cpu') ))
    
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],1800)    
    NetC.load_state_dict(torch.load( Path1+r'/Final_ulteraNetC.pth', map_location=torch.device('cpu') ))

    NetAout=NetA.myforward(phix)
    NetCinp=torch.cat((phit,NetAout),1)
    NetCout=NetC.myforward(NetCinp)
    net_target=NetB.myforward(NetCout)
  
    alltargetcaptions=[d['TargetCaption'] for d in AllData]
    
    #data set images removed duplicates
    if (run_test==0):
        with open(Path1+r'/ultra_unique_query_phix_18.txt', 'rb') as fp:
            phix= pickle.load( fp)
        with open(Path1+r'/ultra_unique_query_img_captions_text.txt', 'rb') as fp:
            allquerycaptions=pickle.load( fp)
        
    else:
        with open(Path1+r'/ultra_unique_query_phix_18_test.txt', 'rb') as fp:
            phix= pickle.load( fp)
        with open(Path1+r'/ultra_unique_query_img_captions_text_test.txt', 'rb') as fp:
            allquerycaptions=pickle.load( fp)

    phix=np.array(phix)
    
    nn_result = []
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    for i in range(phix.shape[0]):
        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])

    for i in range(net_target.shape[0]): 
        sims = net_target[i, :].dot(phix.T) #[:net_target.shape[0],:]
        nn_result.append(np.argsort(-sims[ :])[:110])  
 
    out = []
    nn_result = [[allquerycaptions[nn] for nn in nns] for nns in nn_result]
  
    for k in [1, 5, 10, 50, 100]:    
        r = 0.0
        for i, nns in enumerate(nn_result):
            if alltargetcaptions[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
        out.append(str(k) + ' ---> '+ str(r*100))
        r = 0.0

    print (out)

def Semantic50_5(run_test):
    if run_test==0:
        with open(Path1+r'/FeaturesToFiles172/Features172QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)
        AllData=AllData[:10000]
    else:
        with open(Path1+r'/FeaturesToFiles33/Features33QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)


    phix=[d['Query50F'] for d in AllData]
    phit=[d['ModF'] for d in AllData]
    phix=torch.tensor(phix)
    phit=torch.tensor(phit)
    phit=torch.squeeze(phit)
    target=[d['Target50F'] for d in AllData]
    target=torch.tensor(target)

    
    NetA=NLR3T(phix.shape[1],phit.shape[1],1000)
    NetA.load_state_dict(torch.load( Path1+r'/UltraNetA50tuneCOSVD.pth', map_location=torch.device('cpu') ))
    
    NetB=NLR3T(phit.shape[1],phix.shape[1],2500)
    NetB.load_state_dict(torch.load( Path1+r'/UltraNetB50_CO2704final3_23.pth', map_location=torch.device('cpu') ))
    
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],1800)
    NetC.load_state_dict(torch.load( Path1+r'/Final_ulteraNetC.pth', map_location=torch.device('cpu') ))

    NetAout=NetA.myforward(phix)
    NetCinp=torch.cat((phit,NetAout),1)
    NetCout=NetC.myforward(NetCinp)
    net_target=NetB.myforward(NetCout)
   
    alltargetcaptions=[d['TargetCaption'] for d in AllData]
    
    if (run_test==0):
        with open(Path1+r'/ultra_unique_query_phix_50.txt', 'rb') as fp:
            phix= pickle.load( fp)
        with open(Path1+r'/ultra_unique_query_img_captions_text.txt', 'rb') as fp:
            allquerycaptions=pickle.load( fp)

    else:
        with open(Path1+r'/ultra_unique_query_phix_50_test.txt', 'rb') as fp:
            phix= pickle.load( fp)
        with open(Path1+r'/ultra_unique_query_img_captions_text_test.txt', 'rb') as fp:
            allquerycaptions=pickle.load( fp)
    phix=np.array(phix)

    

    
    nn_result = []
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    for i in range(phix.shape[0]):

        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])

    for i in range (net_target.shape[0]):  
        sims = net_target[i, :].dot(phix.T) #[:net_target.shape[0],:]
        nn_result.append(np.argsort(-sims[ :])[:110])
  
    
    out = []
    nn_result = [[allquerycaptions[nn] for nn in nns] for nns in nn_result]
    
    flags=np.zeros((50))  
    for k in [1, 5, 10, 50, 100]:
    
        r = 0.0
        for i, nns in enumerate(nn_result):
            if alltargetcaptions[i] in nns[:k]:
                if (k==5 and i<50):
                    flags[i]=1
                r += 1
        r /= len(nn_result)
      
        out.append(str(k) + ' ---> '+ str(r*100))
        r = 0.0
    #print(flags)
    print (out)

def Semantic152_5(run_test):
    if run_test==0:
        with open(Path1+r'/FeaturesToFiles172/Features172QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)
        AllData[:10000]
    else:
        with open(Path1+r'/FeaturesToFiles33/Features33QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)

    phix=[d['Query152F'] for d in AllData]
    phit=[d['ModF'] for d in AllData]
    phix=torch.tensor(phix)
    phit=torch.tensor(phit)
    phit=torch.squeeze(phit)
    target=[d['Target152F'] for d in AllData]
    target=torch.tensor(target)
   
    
    NetA=NLR3T(phix.shape[1],phit.shape[1],1000)
    NetA.load_state_dict(torch.load( Path1+r'/UltraNetA152tunesntc.pth', map_location=torch.device('cpu') ))
    
    NetB=NLR3T(phit.shape[1],phix.shape[1],2500)
    NetB.load_state_dict(torch.load( Path1+r'/UltraNetB152_CO25042final2.pth', map_location=torch.device('cpu') ))
    
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],1800)
    NetC.load_state_dict(torch.load( Path1+r'/Final_ulteraNetC.pth', map_location=torch.device('cpu') ))

    NetAout=NetA.myforward(phix)
    NetCinp=torch.cat((phit,NetAout),1)
    NetCout=NetC.myforward(NetCinp)
    net_target=NetB.myforward(NetCout)
 
    alltargetcaptions=[d['TargetCaption'] for d in AllData]
   
    if (run_test==0):
        with open(Path1+r'/ultra_unique_query_phix_152.txt', 'rb') as fp:
            phix= pickle.load( fp)
        with open(Path1+r'/ultra_unique_query_img_captions_text.txt', 'rb') as fp:
            allquerycaptions=pickle.load( fp)

    else:
        with open(Path1+r'/ultra_unique_query_phix_152_test.txt', 'rb') as fp:
            phix= pickle.load( fp)
        with open(Path1+r'/ultra_unique_query_img_captions_text_test.txt', 'rb') as fp:
            allquerycaptions=pickle.load( fp)
    phix=np.array(phix)

    

    nn_result = []
    #phixN=torch.tensor(phixN)
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    for i in range(phix.shape[0]):
        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])

    for i in range (net_target.shape[0]):  #(3900): #
        sims = net_target[i, :].dot(phix.T)  #[:net_target.shape[0],:]
        #print(i)
        nn_result.append(np.argsort(-sims[ :])[:110])
  
    
  # compute recalls
    out = []
    nn_result = [[allquerycaptions[nn] for nn in nns] for nns in nn_result]
  
  
    for k in [1, 5, 10, 50, 100]:
    
        r = 0.0
        for i, nns in enumerate(nn_result):
            if alltargetcaptions[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
   
        out.append(str(k) + ' ---> '+ str(r*100))
        r = 0.0

    print (out)





if __name__ == '__main__':
  Semantic18_5(0)

    
    

    
   

    

