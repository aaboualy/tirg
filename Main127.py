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
#################  Support Functions Section   #################

def dataset(batch_size_all):
    trainset = Fashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        ]))
        

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_all,
                                            shuffle=False, num_workers=2)

    
    return trainset,trainloader

def euclideandistance(signature,signatureimg):
    from scipy.spatial import distance
    return distance.euclidean(signature, signatureimg)
    #.detach().numpy()

def testvaluessame():

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

  transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        ])

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in train.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()

  
  query='women/tops/blouses/91422080/91422080_0.jpeg'
  qttext='replace sunrise with pleat-neck'
  target='women/tops/sleeveless_and_tank_tops/90068628/90068628_0.jpeg'
  
  
  text=[]
  text.append(qttext)
  text.append(qttext)
  

  img = Image.open(Path1+'/'+query)      
  img = img.convert('RGB') 
  img=transform(img)

  img2 = Image.open(Path1+'/'+target)      
  img2 = img2.convert('RGB') 
  img2=transform(img2)

  img=img.unsqueeze_(0)
  img2=img2.unsqueeze_(0)
  images=torch.cat([img, img2], dim=0)

  trigdataQ=trig.compose_img_text(images,text)
  trigdataQ1=trig.compose_img_text(images,text)
  print('...........')
  print(trigdataQ)
  print(trigdataQ1)

def getbetatrainNot():
  
  
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

def GetValuestrain15time():
  
  with open (Path1+"/trainBetaNormalized.txt", 'rb') as fp:
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
  
  trainloader = trainset.get_loader(
          batch_size=2,
          shuffle=True,
          drop_last=True,
          num_workers=0)

  testset = TestFashion200k(
        path=Path1,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trig= TIRG([t.encode().decode('utf-8') for t in trainset.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'

  Results=[]
  
  for i in range(15):
    for name, dataset in [ ('train', trainset)]:  #,('test', testset)]: 
      
      # betaNor="['1 ---> 5.27', '5 ---> 14.39', '10 ---> 21.6', '50 ---> 43.830000000000005', '100 ---> 55.33']"
      # Results.append('No.'+str(i)+' DataSet='+name+' Type= BetaNormalized '+' Result=' +betaNor)
      try:
        
        betaNor = test_retrieval.testbetanormalizednot(opt, trig, dataset,BetaNormalize)
        print(name,' BetaNormalized: ',betaNor)
        Results.append('No.'+str(i)+' DataSet='+name+' Type= BetaNormalized '+' Result=' +betaNor)
      except:
        print('ERROR')

      try:
        asbook = test_retrieval.test(opt, trig, dataset)
        print(name,' As PaPer: ',asbook)
        Results.append('No.'+str(i)+' DataSet='+name+' Type= As PaPer '+' Result=' +betaNor)
      except:
        print('ERROR')

  with open(Path1+r"/"+'Results15time.txt', 'wb') as fp:
    pickle.dump(Results, fp)
      
def distanceBetaand():
  with open (Path1+"/Beta.txt", 'rb') as fp:
    Beta = pickle.load(fp) 

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

  trig= img_text_composition_models.TIRG([t.encode().decode('utf-8') for t in trainset.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()

  imgs = []
  mods = []
  target = []
  batchsize=2
  Distance=[]
  sourceid=[]
  targetid=[]
  countbeta=0
  counttrig=0
  
  for Data in tqdm(trainset):
        
        imgs += [Data['source_img_data']]
        mods += [Data['mod']['str']]
        target +=[Data['target_img_data']]
        sourceid.append(Data['source_img_id'])
        targetid.append(Data['target_img_id'])
   


        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        
        f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

        target = torch.stack(target).float()
        target = torch.autograd.Variable(target)
        f2 = trig.extract_img_feature(target).data.cpu().numpy()

        trigdata=f[0]
        trigbeta = np.insert(trigdata,0, 1)
        trigbeta=np.matmul(trigbeta,Beta) 
        Targetdata = f2[0]

        SourceTarget=euclideandistance(trigdata,Targetdata)
        betaTarget=euclideandistance(trigbeta,Targetdata)

        if(SourceTarget > betaTarget):
            countbeta= countbeta+1
        else:
            counttrig=counttrig+1

    
        # opsig={'source':sourceid[0],'target':targetid[0],'disbeta':betaTarget,'disorig':SourceTarget}
        # Distance.append(opsig )
        
        imgs = []
        mods = []
        target = []
        sourceid=[]
        targetid=[]  
  
  
  with open(Path1+r"/"+'Distance.txt', 'wb') as fp:
    pickle.dump(Distance, fp)

  print('Train Data :Count beta less:',countbeta , ' ,countbeta bigger:',counttrig)

  imgs = []
  mods = []
  target = []
  batchsize=2
  Distance=[]
  sourceid=[]
  targetid=[]
  countbeta=0
  counttrig=0

  for Data in tqdm(test.get_test_queries()):
    imgs += [test.get_img(Data['source_img_id'])]
    mods += [Data['mod']['str']]
    target +=[test.get_img(Data['target_id'])]
    
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    f2 = trig.extract_img_feature(target).data.cpu().numpy()
    trigdata=f[0]
    trigbeta = np.insert(trigdata,0, 1)
    trigbeta=np.matmul(trigbeta,Beta) 
    Targetdata = f2[0]

    SourceTarget=euclideandistance(trigdata,Targetdata)
    betaTarget=euclideandistance(trigbeta,Targetdata)

    if(SourceTarget > betaTarget):
        countbeta= countbeta+1
    else:
        counttrig=counttrig+1

    imgs = []
    mods = []
    target = []
    sourceid=[]
    targetid=[]  

  print('Test Data :Count beta less:',countbeta , ' ,countbeta bigger:',counttrig)

def savesourcevalues():

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
  #trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  #trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'

  #datasets.Features172K().SavetoFilesImageSource(Path1+r'/dataset172', trig, train,opt)
  #datasets.Features33K().SavetoFiles2(Path1+r'/dataset33', trig, test,opt)
  #datasets.Features33K().SavetoFiles3(Path1+r'/dataset33', trig, test,opt)

  #datasets.Features172K().SavetoFilesold(Path1+r'/dataset172', trig, train,opt)
  #datasets.Features33K().SavetoFilesold(Path1+r'/dataset33', trig, test,opt)
  
  datasets.Features172K().SavetoFilesNoModule(Path1+r'/dataset172', trig, train,opt)
  datasets.Features33K().SavetoFilesNoModule(Path1+r'/dataset33', trig, test,opt)
  

  print('172 Finished')
  print('33k Finished')

def savesourcephixtvalues():

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
  #trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'


  #datasets.Features172K().SavetoFiles(Path1+r'/dataset172', trig, train,opt)
  #datasets.Features172K().SavetoFilesImageSource(Path1+r'/dataset172', trig, train,opt)
  #datasets.Features33K().SavetoFiles2(Path1+r'/dataset33', trig, test,opt)
  #datasets.Features33K().SavetoFiles3(Path1+r'/dataset33', trig, test,opt)

  #datasets.Features172K().SavetoFilesphixt(Path1+r'/dataset172', trig, train,opt)
  #datasets.Features33K().SavetoFilesphixt(Path1+r'/dataset33', trig, test,opt)

  # datasets.Features172K().SavetoFilesCaptionImages(Path1+r'/dataset172', trig, train,opt)
  # datasets.Features33K().SavetoFilesCaptionImages(Path1+r'/dataset33', trig, test,opt)

  datasets.Features172K().SavetoFilesphixt(Path1+r'/dataset172Org', trig, train,opt)
  datasets.Features33K().SavetoFilesphixt(Path1+r'/dataset33Org', trig, test,opt)
  
  print('172 Finished')
  print('33k Finished')


#################   get losses from NLP mean & Cosine Section   #################

def getlossesMeanSquare():
  with open (Path1+"\\17720\\loosses.pkl", 'rb') as fp:
    lossMean = pickle.load(fp) 
  
  Y=[]
  X1=[]
  for x in range(len(lossMean)):
    if x % 1 == 0:
      Y.append(lossMean[x].detach().numpy())
      X1.append(x)
  

  plt.plot(X1, Y)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Mean Square Loss Graph')
  plt.show()

def getlossesCos():
  with open (Path1+"\\bk\\17720\\Cos\\loosses2.pkl", 'rb') as fp:
    lossMean = pickle.load(fp) 
  
  Y=[]
  X1=[]
  for x in range(len(lossMean)):
    if x % 1 == 0:
      Y.append(lossMean[x].detach().numpy())
      X1.append(x)
  

  plt.plot(X1, Y)
  # plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
  # plt.rc('ytick', labelsize=30)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Cos Loss Graph')
  plt.show()


#################   Regression Beta created by train datset Section   #################

def getbetatrainLoaded():
    
  imgdata = datasets.Features172K().Get_all_images()
  trigdata = datasets.Features172K().Get_all_queries()
  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)

  Ntrig2=[]
 
  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  for i in range(trigdata.shape[0]):
    Ntrig2.append(np.insert(trigdata[i],0, 1))

  #print("Ntrig2 shape %d  first elemnt %d",Ntrig2[0] )
  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,imgdata) 

  

  with open(Path1+r"/"+'BetatrainLoaded.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestrainloaded():
  
  with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
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
    
    betaNor = test_retrieval.testLoadedBeta(opt, trig, dataset,BetaNormalize)
    print(name,' BetaNormalized: ',betaNor)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)

def getErrorValue():
  with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
    beta = pickle.load(fp) 

  imgdata = datasets.Features172K().Get_all_images()
  all_queries1 = datasets.Features172K().Get_all_queries()
  Error=[]
  #Error1=[]
  for j in range(len(all_queries1)): 
      all_queries1[j, :] /= np.linalg.norm(all_queries1[j, :])
      imgdata[j, :] /= np.linalg.norm(imgdata[j, :])
      Y=imgdata[j]
      X0=all_queries1[j]
      X1 = np.insert(X0,0, 1)
      X2=np.matmul(X1,beta) 
      # Error1=Y-X2
      # if len(Error) ==0:
      #   Error=Error1
      Error.append(Y-X2)
      #Error=(Error+Error1)/2
      # Error1=[]
      # print('X0:',euclideandistance(X0,Y))
      # print('X2:',euclideandistance(X2,Y))
      # print('XE:',euclideandistance(X2 +Error ,Y))
      # print('No:',j)

  Error=np.array(Error)
  Error1=Error.transpose()
  
  with open(Path1+r"/"+'ERRORtrainLoaded.txt', 'wb') as fp:
    pickle.dump(np.matmul(Error1,Error) , fp)

def getbetatrainLoadedold():
    
  imgdata = datasets.Features172K().Get_all_imagesold()
  trigdata = datasets.Features172K().Get_all_queriesold()
  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)

  Ntrig2=[]
 
  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  for i in range(trigdata.shape[0]):
    Ntrig2.append(np.insert(trigdata[i],0, 1))

  #print("Ntrig2 shape %d  first elemnt %d",Ntrig2[0] )
  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,imgdata) 

  

  with open(Path1+r"/"+'BetatrainLoadedold.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestrainloadedold():
  
  with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
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
  #trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
    
    betaNor = test_retrieval.testLoadedBetaold(opt, trig, dataset,BetaNormalize)
    print(name,' BetaNormalized: ',betaNor)

    #asbook = test_retrieval.testLoaded(opt, trig, dataset)
    #print(name,' As PaPer: ',asbook)





#################   Regression Beta created by test datset Section  #################

def getbetatestLoaded():
    
  imgdata = datasets.Features33K().Get_all_target()
  trigdata = datasets.Features33K().Get_all_queries()
  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)

  Ntrig2=[]
 
  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  for i in range(trigdata.shape[0]):
    Ntrig2.append(np.insert(trigdata[i],0, 1))

  #print("Ntrig2 shape %d  first elemnt %d",Ntrig2[0] )
  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,imgdata) 

  

  with open(Path1+r"/"+'BetatestLoaded2.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)



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

  with open(Path1+r"/"+'test_all_source_captionsG.pkl', 'wb') as fp:
    pickle.dump(all_source_captions, fp)
  with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'wb') as fp:
    pickle.dump(all_target_captions, fp)
 

  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)
  with open(Path1+r"/"+'test_all_queriesG.pkl', 'wb') as fp:
    pickle.dump(trigdata, fp)

  with open(Path1+r"/"+'test_all_imgsG.pkl', 'wb') as fp:
    pickle.dump(imgdata, fp)
 
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

  


  with open(Path1+r"/"+'BetatestLoaded.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestestloaded():
  
  with open (Path1+"\\BetatestLoaded2.txt", 'rb') as fp:
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
    
    betaNor = test_retrieval.testLoadedBeta(opt, trig, dataset,BetaNormalize)
    print(name,' BetaNormalized: ',betaNor)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)




#################   Regression Beta created by avg between two dataset beta Section  #################

def GetAverageBeta():
  with open (Path1+"/BetatrainLoaded.txt", 'rb') as fp:
    BetaTrain = pickle.load(fp) 

  with open (Path1+"/BetatestLoaded2.txt", 'rb') as fp:
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
    
    betaNor = test_retrieval.testLoadedBeta(opt, trig, dataset,BetaAvg2)
    print(name,' Beta Avg: ',betaNor)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)


#################   Regression Beta created by test & train datset Section  #################

def getbetatrainandtestLoaded():
    
  imgdata1 = datasets.Features172K().Get_all_images()
  trigdata1 = datasets.Features172K().Get_all_queries()
  imgdata2 = datasets.Features33K().Get_all_target()
  trigdata2 = datasets.Features33K().Get_all_queries()
  
  # for i in range(trigdata2.shape[0]):
  #   trigdata+= [trigdata2[i]]

  # for i in range(imgdata2.shape[0]):
  #   imgdata+= [imgdata2[i]]
   
  trigdata0=[]
  trigdata0.append(trigdata1)
  trigdata0.append(trigdata2)

  imgdata0=[]
  imgdata0.append(imgdata1)
  imgdata0.append(imgdata2)

  imgdata = [item for sublist in imgdata0 for item in sublist]
  trigdata = [item for sublist in trigdata0 for item in sublist]

  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)

  Ntrig2=[]
 
  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  for i in range(trigdata.shape[0]):
    Ntrig2.append(np.insert(trigdata[i],0, 1))

  #print("Ntrig2 shape %d  first elemnt %d",Ntrig2[0] )
  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,imgdata) 

  

  with open(Path1+r"/"+'BetatraintestLoaded.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestrainandtestloaded():
  
  with open (Path1+"\\BetatraintestLoaded.txt", 'rb') as fp:
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
    
    betaNor = test_retrieval.testLoadedBeta(opt, trig, dataset,BetaNormalize)
    print(name,' BetaNormalized: ',betaNor)

    asbook = test_retrieval.testLoaded(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)



################# Regression beta from out NLP ################

def getbetaOPNLPLoaded():
    
  imgdata = datasets.Features172K().Get_all_images()
  trigdata = datasets.Features172K().Get_all_queries()

  hidden1=1050
  hidden2=950


  model=NLR2(512,512,hidden1,hidden2)
  model.load_state_dict(torch.load(Path1+r'\NLP41COS172K15000.pth' , map_location=torch.device('cpu') ))
  model.eval()

  trigdata=(torch.Tensor(trigdata))
  trigdata=model.myforward(trigdata).data.cpu().numpy()

  trigdata=np.array(trigdata)
  imgdata=np.array(imgdata)

  Ntrig2=[]
 
  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  for i in range(trigdata.shape[0]):
    Ntrig2.append(np.insert(trigdata[i],0, 1))

  #print("Ntrig2 shape %d  first elemnt %d",Ntrig2[0] )
  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,imgdata) 

  

  with open(Path1+r"/"+'BetaOPNLPLoaded.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

  print('Finished')

def resultsNLPCosWOPbetaNLP():
  
  with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
    OldBeta = pickle.load(fp) 

  with open (Path1+"\\BetaOPNLPLoaded.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 
  
  hidden1=1050
  hidden2=950


  model=NLR2(512,512,hidden1,hidden2)
  model.load_state_dict(torch.load(Path1+r'\NLP41COS172K15000.pth' , map_location=torch.device('cpu') ))
  model.eval()


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
    
    NLP2 = test_retrieval.testLoadedNLPwBeta(opt, trig, dataset, OldBeta, model)
    print(name,' NLP then OLD beta : ',NLP2)

    NLP2 = test_retrieval.testLoadedNLPwBeta(opt, trig, dataset, BetaNormalize, model)
    print(name,' NLP then New beta : ',NLP2)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)



#################   NLP by Mean Square Value Section  #################
class NLR2(nn.Module):
  def __init__(self,netin,netout,nethidden1,nethidden2):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.Sigmoid(),torch.nn.Linear(nethidden1, nethidden2),
    torch.nn.Linear(nethidden2, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

class NLR3(nn.Module):
  def __init__(self,netin,netout,nethidden):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden),torch.nn.Sigmoid(),torch.nn.Linear(nethidden, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

def build_and_train_netMSE():


  hidden1=1050
  hidden2=950
  batch_size=500
  max_iterations=25000
  min_error=0.01
    


  all_queries= datasets.Features172K().Get_all_queries()
  all_imgs=datasets.Features172K().Get_all_images()
  all_queries=Variable(torch.Tensor(all_queries))
  all_imgs=Variable(torch.tensor(all_imgs))

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
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
    
    #for x in range(901, 1100, 2):

      model=NLR2(512,512,hidden1,hidden2)
      #model.load_state_dict(torch.load(Path1+r'\NLPMSE172K900-800'+str(x)+r'.pth' , map_location=torch.device('cpu') ))
      model.load_state_dict(torch.load(Path1+r'\NLPMSEtFinal.pth' , map_location=torch.device('cpu') ))
      model.eval()

      NLP = test_retrieval.testLoadedNLP(opt, trig, dataset, model)
      #print(name,'NLP:',str(x),':-->',NLP)
      print(name,'NLP Mean:',':-->',NLP)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)


#################   NLP by Cosine Section  #################


def build_and_train_netCOS():
  hidden1=1050
  hidden2=950
  batch_size=500
  max_iterations=15000
  min_error=0.01
    


  all_queries= datasets.Features172K().Get_all_queries()
  all_imgs=datasets.Features172K().Get_all_images()
  all_queries=Variable(torch.Tensor(all_queries))
  all_imgs=Variable(torch.tensor(all_imgs))
  model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)
  
  torch.manual_seed(3)
  loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  torch.manual_seed(3)
  
  criterion=nn.CosineSimilarity(dim=1, eps=1e-6)


  optimizer=torch.optim.SGD(model.parameters(), lr=0.002)
  epoch=max_iterations

  losses=[]
  for j in range(epoch):
    total_loss=0
    for l in range(int(all_queries.shape[0]/batch_size)):
      
      item_batch = all_queries[l*batch_size:(l+1)*batch_size-1,:]
      target_batch=all_imgs[l*batch_size:(l+1)*batch_size-1,:]
      netoutbatch=model.myforward(item_batch)
      #loss = loss_fn(target_batch,netoutbatch)
      loss = torch.mean(torch.abs(1-loss_fn(target_batch,netoutbatch)))

      #loss=1-loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
    if (total_loss<min_error):
      break
    print('iteration:',j, 'total loss=',total_loss,'last batch loss=',loss)
    losses.append(total_loss)
    if(j%1000==0):
      torch.save(model.state_dict(), Path1+r'\NLP3COS172K'+str(j)+'.pth') 

  print ('mean square loss',loss_fn(model.myforward(all_queries),all_imgs))  
  print('Finished Training')
  with open(Path1+r"/"+'loosses3.pkl', 'wb') as fp:
          pickle.dump( losses, fp)

  torch.save(model.state_dict(), Path1+r'\NLP3COSfinal172k.pth') 


def resultsNLPCos():
  
  
  hidden1=1050
  hidden2=950


  model=NLR2(512,512,hidden1,hidden2)
  model.load_state_dict(torch.load(Path1+r'\NLP41COS172K15000.pth' , map_location=torch.device('cpu') ))
  model.eval()


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
    
    NLP = test_retrieval.testLoadedNLP(opt, trig, dataset, model)
    print(name,' NLP cos: ',NLP)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)

#################   Regression  beat Train  & NLP by Cosine Section  #################


def resultsNLPCosWbeta():
  
  with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 
  
  hidden1=1050
  hidden2=950


  model=NLR2(512,512,hidden1,hidden2)
  model.load_state_dict(torch.load(Path1+r'\NLP41COS172K15000.pth' , map_location=torch.device('cpu') ))
  model.eval()


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
    
    NLP = test_retrieval.testLoadedBetaWNLP(opt, trig, dataset, BetaNormalize, model)
    print(name,' beta then NLP: ',NLP)

    NLP2 = test_retrieval.testLoadedNLPwBeta(opt, trig, dataset, BetaNormalize, model)
    print(name,' NLPthen beta : ',NLP2)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)


###################3172021 #########################


def LinearModel():
  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   beta = pickle.load(fp) 

  imgdata = datasets.Features172K().Get_all_images()
  all_queries1 = datasets.Features172K().Get_all_queries()

  reg = LinearRegression().fit(all_queries1, imgdata)
  print(reg.intercept_)
  print('')
  #value=reg.score(all_queries1, imgdata)
  #print(value)
  #print(reg.coef_)
  # X0=all_queries1[0]
  # valuesX=[]
  # valuesY=[]
  # for j in range(len(all_queries1)):
    
  #   Y=imgdata[j]

  #   X1 = np.insert(X0,0, 1)
  #   X2=np.matmul(X1,beta) 

  #   print('X0:',euclideandistance(X0,Y))
  #   print('X0Beta:',euclideandistance(X2,Y))
  #   X0dash=[X0]#np.reshape(X0, (0, 1))
  #   print('X0dash:',euclideandistance(reg.predict(X0dash),Y))
  #   print('')
    

  # plt.plot(X1, Y)
  # plt.xlabel('Iteration')
  # plt.ylabel('Loss')
  # plt.title('Mean Square Loss Graph')
  # plt.show()

def GetValuesRegModel():

  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  imgdata = datasets.Features172K().Get_all_images()
  all_queries1 = datasets.Features172K().Get_all_queries()

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
    
    #betaNor = test_retrieval.testLoadedBetaRegModel(opt, trig, dataset,Beta,reg)
    # betaNor = test_retrieval.testLoadedRegModel(opt, trig, dataset,reg)
    # print(name,' BetaNormalized: ',betaNor)

    asbook = test_retrieval.test(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)

def GetValuesRandomForestRegressor():
  SizeI=50000
  n_estimator=50
  max_depths=25

  imgdata = datasets.Features172K().Get_all_images()[:SizeI]
  all_queries1 = datasets.Features172K().Get_all_queries()[:SizeI]

  for i in range(all_queries1.shape[0]):
    all_queries1[i, :] /= np.linalg.norm(all_queries1[i, :])
  for i in range(imgdata.shape[0]):
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

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)

################### MeanSquareERROR #########################

def getlossesMeanSquarelinear():

  with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
    beta = pickle.load(fp) 

  all_queries=[]
  all_queriestrain=[]

  imgdatatrain = datasets.Features172K().Get_all_images()
  all_queries1train = datasets.Features172K().Get_all_queries()
  print('Mean Square ERROR of TrainDataset Trig:',mean_squared_error(imgdatatrain,all_queries1train))

  for j in range(len(all_queries1train)): 
      all_queries1train[j, :] /= np.linalg.norm(all_queries1train[j, :])
      X1 = np.insert(all_queries1train[j],0, 1)
      X2=np.matmul(X1,beta) 
     
      all_queriestrain.append(X2)


  print('Mean Square ERROR of TrainDataset Trig Linear:',mean_squared_error(imgdatatrain,all_queriestrain))

  imgdata = datasets.Features33K().Get_all_target()
  all_queries1 = datasets.Features33K().Get_all_queries()
  print('Mean Square ERROR of TestDataset Trig:',mean_squared_error(imgdata,all_queries1))

  for j in range(len(all_queries1)): 
      all_queries1[j, :] /= np.linalg.norm(all_queries1[j, :])
      X1 = np.insert(all_queries1[j],0, 1)
      X2=np.matmul(X1,beta) 
     
      all_queries.append(X2)

  print('Mean Square ERROR of TestDataset Trig Linear:',mean_squared_error(imgdata,all_queries))

  

 

################### Regphix #########################


def GetValuesRegModelphix():

  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  imgdata = datasets.Features172K().Get_phixtarget()
  all_queries1 = datasets.Features172K().Get_phix()

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
    
    #betaNor = test_retrieval.testLoadedBetaRegModel(opt, trig, dataset,Beta,reg)
    betaNor = test_retrieval.testLoadedRegModelphix(opt, trig, dataset,reg)
    print(name,' Reg: ',betaNor)

    # asbook = test_retrieval.test(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)

def GetValuesRandomForestRegressorphix():
  imgdata = datasets.Features172K().Get_phixtarget()
  all_queries1 = datasets.Features172K().Get_phix()

  for i in range(all_queries1.shape[0]):
    all_queries1[i, :] /= np.linalg.norm(all_queries1[i, :])
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])

  print('1')
  regr = RandomForestRegressor(max_depth=50, random_state=0)
  print('2')
  regr.fit(all_queries1, imgdata)
  print('3')
  

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
    
    betaNor = test_retrieval.testLoadedRandomForestRegressorphix(opt, trig, dataset,regr)
    print(name,' Random: ',betaNor)

    # asbook = test_retrieval.testLoaded(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)




######################  RUN  ##########################

# train  As PaPer:  ['1 ---> 33.21', '5 ---> 60.67', '10 ---> 73.18', '50 ---> 92.97999999999999', '100 ---> 97.19']
# test  As PaPer:  ['1 ---> 14.04121863799283', '5 ---> 34.268219832735966', '10 ---> 42.41338112305854', '50 ---> 65.31660692951016', '100 ---> 73.66487455197132']





######################### 2882021 ########################
class ConNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.netmodel= torch.nn.Sequential(nn.Conv2d(1,3, kernel_size=(2,2)),nn.Conv2d(3,7, kernel_size=(2,2),stride=1))
    self.f1=nn.Linear( 6300,512)
  def myforward (self,inv):
    outv=self.netmodel(inv)
    outv = outv.view(outv.size(0), -1)
    outv=self.f1(outv)
    return outv

class ConNet_img_text(nn.Module):
  def __init__(self):
    super().__init__()
    self.netmodel= torch.nn.Sequential(nn.Conv1d(1,2, 3,stride=1),nn.Conv1d(2,3, kernel_size=3,stride=1),nn.MaxPool1d(4))
    self.l1=nn.Linear( 893,1024)
    self.l2=nn.Linear(1024,512)
  def myforward (self,inv):
    outv=self.netmodel(inv)
    outv = outv.view(outv.size(0), -1)
    invsq= torch.squeeze(inv)
    sz=outv.size()
    szinpv=inv.size() 
    tmp=torch.zeros([sz[0],sz[1]+szinpv[2]])
    tmp[:,0:sz[1]]=outv
    tmp[:,sz[1]:sz[1]+szinpv[2]]=invsq
    outv=tmp
    outv=self.l2(torch.sigmoid(self.l1(outv)))
    return outv

class ConNet_text(nn.Module):
  def __init__(self):
    super().__init__()
    self.netmodel= torch.nn.Sequential(nn.Conv1d(1,2, 3,stride=1),nn.Conv1d(2,3, kernel_size=3,stride=1),nn.MaxPool1d(3))
    self.l1=nn.Linear( 1019,1024)
    self.l2=nn.Linear(1024,512)
  def myforward_text (self,inv):
    outv=self.netmodel(inv)
    outv = outv.view(outv.size(0), -1)
    invsq= torch.squeeze(inv)
    sz=outv.size()
    szinpv=inv.size() 
    tmp=torch.zeros([sz[0],sz[1]+szinpv[2]])
    tmp[:,0:sz[1]]=outv
    tmp[:,sz[1]:sz[1]+szinpv[2]]=invsq
    outv=tmp
    outv=self.l2(torch.sigmoid(self.l1(outv)))
    return outv

class ConNet_img(nn.Module):
  def __init__(self):
    super().__init__()
    self.netmodel= torch.nn.Sequential(nn.Conv1d(1,3, kernel_size=(5)),nn.Conv1d(3,7, kernel_size=(5),stride=1))
    self.l1=nn.Linear( 3528,1024)
    self.l2=nn.Linear(1024,512)

  def myforward_img (self,inv):
    outv=torch.relu(inv)

    outv=self.netmodel(outv)
    outv = outv.view(outv.size(0), -1)
    outv=self.l2(torch.tanh(self.l1(outv)))
    return outv
###########################################training network method #######################333
#######################################################################3333333333333
def phase2_text_img_phi(model,phix,phix_target,iterations,totallosses,batch_size,loss_fn,optimizer,min_error):
  for j in range(iterations):
    total_loss=0
    for l in range(int(phix.shape[0]/batch_size)):
      
      netoutbatch=model.myforward(torch.FloatTensor(phix[l*batch_size:(l+1)*batch_size,:]))
      target_batch=torch.FloatTensor(phix_target[l*batch_size:(l+1)*batch_size,:])
      loss = loss_fn(target_batch,netoutbatch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
              
    if (total_loss<min_error):
      break
    print('iteration:',j, 'total loss=',total_loss,'batch_loss=', loss)
    totallosses.append(total_loss)
  return model,totallosses,total_loss      

############################################################################################
#############################################################################################333

def phase2_two_network_model_train():
  
  phix=datasets.Features172K().Get_phix()
  phit=datasets.Features172K().Get_phit()
  phix_target=datasets.Features172K().Get_phixtarget()
  phix=phix.reshape(phix.shape[0],1,512)
  phit=torch.tensor(phit)
  phit=phit.reshape(phit.shape[0],1,512)

  epoch=10
  iterations=1000
  batch_size=700
  min_error=0.01
  glr=0.003
  first_time_flag=0
  for i in range(epoch):

  #############################################################3
  ###################img#########################
    print('image network')
    if first_time_flag==0:
      model_img=ConNet_img_text()
      torch.manual_seed(300)
      loss_fn = torch.nn.MSELoss()
      optimizer=torch.optim.SGD(model_img.parameters(), lr=glr)
      totallosses_img=[]
    model_img,totallosses_img,total_loss_img=phase2_text_img_phi(model_img,phix,phix_target,iterations,totallosses_img,batch_size,loss_fn,optimizer,min_error)
    print('text network')
    if first_time_flag==0:
      model_text=ConNet_img_text()
      torch.manual_seed(70)
      loss_fn = torch.nn.MSELoss()
      optimizer=torch.optim.SGD(model_text.parameters(), lr=glr)
      totallosses_text=[]
      first_time_flag=1
    model_text,totallosses_text,total_loss_text=phase2_text_img_phi(model_text,phit,phix_target,iterations,totallosses_text,batch_size,loss_fn,optimizer,min_error)
    if (total_loss_img<min_error and total_loss_text<min_error):
      break
    print('Epoch:',i, 'total loss',total_loss_img,total_loss_text)
    torch.save(model_img.state_dict(), Path1+r'\phase2_img'+str(i)+r'.pth') 
    with open(Path1+r"/"+'phase2_losses_img.txt', 'wb') as fp:
        pickle.dump(totallosses_img, fp)
    torch.save(model_text.state_dict(), Path1+r'\phase2_text'+str(i)+r'.pth') 
    with open(Path1+r"/"+'phase2_losses_text.txt', 'wb') as fp:
        pickle.dump(totallosses_text, fp)

  ###############################end img#########################33#########################
   
def get_joint_results (img_model_file,text_model_file,model, testset,flag):
  model.eval()
  test_queries = testset.get_test_queries()
  
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    
    all_imgs = datasets.Features33K().Get_all_images() #[:20000]
    all_captions = datasets.Features33K().Get_all_captions()
    all_queries = datasets.Features33K().Get_all_queries() #[:20000]
    all_target_captions = datasets.Features33K().Get_target_captions()
    if flag==1:
      phi=datasets.Features33K().Get_phix()
    else:
      if flag==2:
        phi = datasets.Features33K().Get_phit()
      else:
        phi_img=datasets.Features33K().Get_phix()
        phi_text=datasets.Features33K().Get_phit()


    if flag!=3:
      phi=phi.reshape(phi.shape[0],1,512)
    else:
      phi_img=phi_img.reshape(phi_img.shape[0],1,512)
      phi_text = np.concatenate(phi_text)
      phi_text=phi_text.reshape(phi_text.shape[0],1,512)


    if flag==3:
      new_all_queries_img=phase2_models_single(phi_img,img_model_file,1)
      new_all_queries_text=phase2_models_single(phi_text,text_model_file,2)

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images()[:10000]
    
    all_captions = datasets.Features172K().Get_all_captions()[:10000]
    all_queries = datasets.Features172K().Get_all_queries()[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions()  #[:10000]
    if flag==1:
      phi = datasets.Features172K().Get_phix()
    if flag ==2:
      phi= datasets.Features172K().Get_phit()
      phi=phi.reshape(phi.shape[0],1,512)
      new_all_queries_img=phase2_models_single(phi,img_model_file,flag)

    if flag==3:
      phi_img = datasets.Features172K().Get_phix()[:10000]
      phi_text= datasets.Features172K().Get_phit()[:10000]
      phi_img=phi_img.reshape(phi_img.shape[0],1,512)
      phi_text = np.concatenate(phi_text)
      phi_text=phi_text.reshape(phi_text.shape[0],1,512)
      new_all_queries_img=phase2_models_single(phi_img,img_model_file,1)
      new_all_queries_text=phase2_models_single(phi_text,text_model_file,2)

  # feature normalization
  
  for i in range(all_queries.shape[0]):
    new_all_queries_img[i, :] /= np.linalg.norm(new_all_queries_img[i, :])
    new_all_queries_text[i, :] /= np.linalg.norm(new_all_queries_text[i, :])

  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result_img = []
  nn_result_text = []
  joint_result=[]
  for i in tqdm(range(all_queries.shape[0])):
    sims_img = new_all_queries_img[i:(i+1), :].dot(all_imgs.T)
    sims_text = new_all_queries_text[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims_img[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      sims_text[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
    temp_img=np.argsort(-sims_img[0, :])[:105]
    temp_text=np.argsort(-sims_text[0, :])[:105]
    nn_result_img.append(temp_img)
    nn_result_text.append(temp_text)
    joint_result.append(get_joint_list(temp_img,temp_text))

    nn_result_img.append(np.argsort(-sims_img[0, :])[:105])
    nn_result_text.append(np.argsort(-sims_text[0, :])[:105])
    #joint_result=get_joint_list(nn_result_img,nn_result_text)
  # compute recalls
  out = []
  out2=[]
  out3=[]
  with open(Path1+r"/"+'jointresultsrecallimages.txt', 'wb') as fp:
        pickle.dump(joint_result, fp)
  nn_result_img = [[all_captions[nn] for nn in nns] for nns in nn_result_img]
  nn_result_text = [[all_captions[nn] for nn in nns] for nns in nn_result_text]
  joint_result = [[all_captions[nn] for nn in nns] for nns in joint_result]


  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(joint_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(joint_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out3.append(str(k) + ' ---> '+ str(r*100))

    
    r = 0.0
    for i, nns in enumerate(nn_result_img):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result_img)
    #out2 += [('recall_top' + str(k) + '_correct_composition', r)]
    out2.append(str(k) + ' ---> '+ str(r*100))

    r = 0.0
    for i, nns in enumerate(nn_result_text):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result_text)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out.append(str(k) + ' ---> '+ str(r*100))

  print('text:', out )
  print('cmi:', out2 )
  
  return out, out2, out3
def get_joint_list(nn_result_img,nn_result_text):
  joint_list_temp=[]
  ln=0
  required_len=nn_result_img.shape[0]
  for i in range(nn_result_img.shape[0]):
    if nn_result_img[i] in nn_result_text:
      joint_list_temp.append(nn_result_img[i])
      ln +=1
      if ln>=required_len:
        break
  if len(joint_list_temp)<required_len:
    for i in range(nn_result_img.shape[0]):
      if nn_result_img[i] not in joint_list_temp:
        joint_list_temp.append(nn_result_img[i])
        ln +=1
        if ln>=required_len:
          break
  return joint_list_temp
def phase2_network_combined_one():
  phix = datasets.Features33K().Get_phix()
  phit = datasets.Features33K().Get_phit()
  phitarget = datasets.Features33K().Get_all_target()
  phit = np.concatenate(phit)
  print(phitarget.shape[0])

  W1=1
  W2=2
  epoch=10000
  batch_size=500
  min_error=0.01
  glr=0.006
  phix=phix*W2
  phit=phit* W1
  combinedxt=[]
  

  for i in range(phix.shape[0]):
    combinedxt.append([ [y for x in [phit[i], phix[i]] for y in x]])

  model=ConNet()
  torch.manual_seed(3)
  loss_fn = torch.nn.MSELoss()
  torch.manual_seed(3)
  criterion=nn.MSELoss()
  optimizer=torch.optim.SGD(model.parameters(), lr=glr)
  
  losses=[]
  totallosses=[]

  combinedxt1024 = np.concatenate(combinedxt)
  combinedxt32=combinedxt1024.reshape(phix.shape[0],1,32,32)
  for j in range(epoch):
    total_loss=0
    for l in range(int(combinedxt32.shape[0]/batch_size)):
      
      # for l in range(int(50000/batch_size)):      
      # item_batch = all_queries[l*batch_size:(l+1)*batch_size-1,:]
      # target_batch=all_imgs[l*batch_size:(l+1)*batch_size-1,:]

      #combinedxt1024 = np.concatenate(combinedxt[l])
      #combinedxt32=combinedxt1024.reshape(1,1,32,32)
      netoutbatch=model.myforward(torch.FloatTensor(combinedxt32[l*batch_size:(l+1)*batch_size,:]))
      target_batch=torch.FloatTensor(phitarget[l*batch_size:(l+1)*batch_size,:])
      
      loss = loss_fn(target_batch,netoutbatch)
      losses.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
      if (l%1000==0) :
        print('Epoch:',j,' get images batch=',l,'loss',loss,end='\r')
        
    if (total_loss<min_error):
      break
    print('iteration:',j, 'total loss',total_loss)
    totallosses.append(total_loss)
    if (j%1000==0) :
       torch.save(model.state_dict(), Path1+r'\3lr006w12'+str(j)+r'.pth') 
       with open(Path1+r"/"+'losses_2.txt', 'wb') as fp:
        pickle.dump(totallosses, fp)
  #print ('NewModel:',loss_fn(model.myforward(all_queries),all_queries))  
  print('Finished Training')
  torch.save(model.state_dict(), Path1+r'\2CovLinearflr006w12.pth') 

def test_model(file_name):
   test_model=ConNet()
   test_model.load_state_dict(torch.load(Path1+r'/'+file_name , map_location=torch.device('cpu') ))
   test_model.eval()
   phix = datasets.Features172K().Get_phix()
   phit = datasets.Features172K().Get_phit()
   phitarget = datasets.Features172K().Get_phixtarget()
   phix=phix[0:100000,:]
   phit = np.concatenate(phit)

   phit=phit[0:100000,:]
   phitarget=phitarget[0:100000,:]

   W1=1
   W2=2
   phix=phix*W2
   phit=phit* W1
   combinedxt=[phit,phix]
   combinedxt1024 = np.concatenate(combinedxt,axis=1)
   combinedxt32=combinedxt1024.reshape(phix.shape[0],1,32,32)
   combinedxt.clear()
   combinedxt1024=[]
   combinedxt32=torch.FloatTensor(combinedxt32)
   net_combined=test_model.myforward(combinedxt32)
   combinedxt32=[]
   net_combined = Variable(net_combined, requires_grad=False)

   net_combined=np.array(net_combined)
   out=test_retrieval.Phase2_networks_tests(1,net_combined)

   combined_all=[phix, phit, net_combined]
   phix=[]
   phit=[]
   
   combined_all=np.concatenate(combined_all,axis=1)
   reg = LinearRegression().fit(combined_all, phitarget)
   new_target=reg.predict(combined_all)
   mse= np.average(np.square((new_target-phitarget)))
   print('mse ',mse)
   out=test_retrieval.Phase2_networks_tests(1,new_target)
   print(out)
   
def Phase2_test_models_get_orignal(img_model_file,text_model_file,flag):

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
  
  #for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset), 
  #for name, dataset in [ ('test', testset)]:
  for name, dataset in [ ('train', trainset)]:
     #('train', trainset), 

    
    #asbook1, model, euc_model = phase2_main_test_gate(opt, trig, dataset,model_file_name,flag)
    asbook1, model = get_joint_results(img_model_file,text_model_file,trig, dataset,flag)
    print(name,' Loaded As PaPer: ',asbook1, '\n  model generated      ',model, '\n  ' )

     

def phase2_main_test_gate(opt, model, testset,model_file_name,flag):
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
    if flag==1:
      phi=datasets.Features33K().Get_phix()
    else:
      phi = datasets.Features33K().Get_phit()
    phi=phi.reshape(phi.shape[0],1,512)

    new_all_queries=phase2_models_single(phi,model_file_name,flag)

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images() #[:10000]
    
    all_captions = datasets.Features172K().Get_all_captions() #[:10000]
    all_queries = datasets.Features172K().Get_all_queries() #[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions() #[:10000]
    if flag==1:
      phi = datasets.Features172K().Get_phix()
    if flag ==2:
      phi= datasets.Features172K().Get_phit()
    phi=phi.reshape(phi.shape[0],1,512)

    new_all_queries=phase2_models_single(phi,model_file_name,flag)


  # feature normalization
  diff=new_all_queries-all_queries
  diff=diff**2
  diff=np.sum(diff,axis=1)
  print(np.mean(diff))

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
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    new_sims = new_all_queries[i:(i+1), :].dot(all_imgs.T)
    euc_new_sims=np.sum(abs(all_imgs-all_queries[i, :]),axis=1)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      new_sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      euc_new_sims[test_queries[i]['source_img_id']]=10e10

    nn_result.append(np.argsort(-sims[0, :])[:110])
    new_nn_result.append(np.argsort(-new_sims[0, :])[:110])
    euc_new_nn_result.append(np.argsort(euc_new_sims)[:110])

  # compute recalls
  out = []
  out2=[]
  out3=[]
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  new_nn_result = [[all_captions[nn] for nn in nns] for nns in new_nn_result]
  euc_new_nn_result = [[all_captions[nn] for nn in nns] for nns in euc_new_nn_result]

  for k in [1, 5, 10, 50, 100]:
    r = 0.0
  
    for i, nns in enumerate(euc_new_nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(euc_new_nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out3.append(str(k) + ' ---> '+ str(r*100))
    
    r = 0.0
    for i, nns in enumerate(new_nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(new_nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out2.append(str(k) + ' ---> '+ str(r*100))

    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out.append(str(k) + ' ---> '+ str(r*100))

    
  return out, out2, out3
def phase2_models_single(phi,model_file_name,flag):
   if flag==1 :
    test_model=ConNet_img()
   else:
    test_model=ConNet_text()
   test_model.load_state_dict(torch.load(Path1+r'\\'+model_file_name , map_location=torch.device('cpu') ))
   test_model.eval()
   phi=phi.reshape(phi.shape[0],1,512)

   phi=torch.FloatTensor(phi)

   if flag==1:
    predict_phi=test_model.myforward_img(phi)
   if flag==2:
     predict_phi=test_model.myforward_text(phi) 
   predict_phi = Variable(predict_phi, requires_grad=False)
   predict_phi=np.array(predict_phi)
   return predict_phi


def phase2_models(phix,phit,model_file_name):
   test_model=ConNet_img()
   test_model.load_state_dict(torch.load(Path1+r'\\'+model_file_name , map_location=torch.device('cpu') ))
   test_model.eval()
   #phix=phix[0:100000,:]
   phit = np.concatenate(phit)
   #phit=phit[0:100000,:]
   W1=1
   W2=2
   phix=phix*W2
   phit=phit* W1
   combinedxt=[phit,phix]
   combinedxt = np.concatenate(combinedxt,axis=1)
   combinedxt=combinedxt.reshape(phix.shape[0],1,32,32)
   combinedxt=torch.FloatTensor(combinedxt)
   net_combined=test_model.myforward(combinedxt)
   net_combined = Variable(net_combined, requires_grad=False)
   net_combined=np.array(net_combined)
   return net_combined

def CBIR(input_id,recall_type, recall_length,model_file,train):
  # test recall function visually
  # input_id   is the image under test sequence number
  # recall_type image, text, combined
  # Model_file if neural model_file
    figcount=7
    fig,ax=plt.subplots(1,figcount)
    display_image (train,input_id,ax,0)
    im=train[input_id]

    target_id =im['target_img_id']
    combined_text='img:'+im['source_caption']+' Target:' + im['target_caption']+' mod:' + im['mod']['str']
    display_image(train,target_id,ax,1)
    fig.suptitle(combined_text)

    model_img=ConNet_text()
    model_img.load_state_dict(torch.load(Path1+r'\\'+model_file , map_location=torch.device('cpu') ))
    all_imgs = datasets.Features172K().Get_all_images() #[:10000]
    all_captions = datasets.Features172K().Get_all_captions() #[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions() #[:10000]
    phix = datasets.Features172K().Get_phix()[input_id]
    phix=phix.reshape(1,1,512)
    
    netout=model_img.myforward_text(torch.FloatTensor(phix))
    #netout=torch.squeeze(netout)
    #phix /= np.linalg.norm(phix)
    
    #for i in range(all_imgs.shape[0]):
    #  all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
    nn_result = []
    netout= torch.autograd.Variable(netout)

    netout=np.array(netout)
    #all_imgs=torch.FloatTensor(all_imgs)
    #netout[0,:]/=np.linalg.norm(netout[0,:])
    #sims = netout[0,:].dot(all_imgs.T)
    #sims=phix[0,0,:].dot(all_imgs.T)
    #nn_result.append(np.argsort(-sims[ :])[:recall_length])
    nn_result=get_distance_feature(all_imgs[input_id,:],all_imgs,512,2,target_id)
    recalled_image_ids=nn_result
    recalled_image_ids=np.array(recalled_image_ids)
    #for i in range(5):
      
    #  display_image (train,recalled_image_ids[i,1],ax,i+2)
  
  # compute recalls
    out = []
    id_list=[]
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    r=0
    plt.show()

    if all_target_captions[input_id] in nn_result[:recall_length]:
      r += 1
    print('recalled list',recalled_image_ids,'Found count=',r, 'Found List=',id_list)
    
    return r
def display_image (dataset,id,ax,i):
  query_img=dataset.get_img(id)
  im=ax[i].imshow(query_img.data.swapaxes(0,1).swapaxes(1,2))



def get_distance_feature(netout,all_imgs,vec_len,req_len_per_feature,target_id):
  nn_result=[]
  min=0
  for i in range (vec_len):
    #sims = netout[0,:].dot(all_imgs.T)
    sims=((all_imgs.T)[i]-netout[i])**2
    tmp=np.argsort(sims[ :])
    index=np.where(tmp==target_id)
    if index[0]<min:
      min=index[0]
    nn_result.append(tmp[:req_len_per_feature])
    #nn_result.append(np.argsort(sims[ :])[:req_len_per_feature])

  print(min)

  return nn_result
  



def select_best_saved_model():
  base_net=ConNet_img_text()
  loss_fn = torch.nn.MSELoss()
  phix=datasets.Features172K().Get_phix()[:20000]
  phit=datasets.Features172K().Get_phit()[:20000]
  phix_target=datasets.Features172K().Get_phixtarget()[:20000]
  phix=phix.reshape(phix.shape[0],1,512)
  phit=torch.tensor(phit)
  phit=phit.reshape(phit.shape[0],1,512)

  for i in range(3):
    base_net.load_state_dict(torch.load(Path1+r'\phase2_img'+str(i)+r'.pth', map_location=torch.device('cpu') ))
    netoutbatch=base_net.myforward(torch.FloatTensor(phix))
    target_batch=torch.FloatTensor(phix_target)
    loss = loss_fn(target_batch,netoutbatch)
    print('loss for model image',i, 'is ', loss)
    base_net.load_state_dict(torch.load(Path1+r'\phase2_text'+str(i)+r'.pth', map_location=torch.device('cpu') ))
    netoutbatch=base_net.myforward(torch.FloatTensor(phit))
    target_batch=torch.FloatTensor(phix_target)
    loss = loss_fn(target_batch,netoutbatch)
    print('loss for model text',i, 'is ', loss)

def save_out_of_conv_step():
  base_net=ConNet_img_text()
  phix=datasets.Features172K().Get_phix()
  phit=datasets.Features172K().Get_phit()
  phix=phix.reshape(phix.shape[0],1,512)
  phit=torch.tensor(phit)
  phit=phit.reshape(phit.shape[0],1,512)

  i=0
  base_net.load_state_dict(torch.load(Path1+r'\phase2_img'+str(i)+r'.pth', map_location=torch.device('cpu') ))
  netoutbatch=base_net.myforward(torch.FloatTensor(phix))
  with open(Path1+r"/"+'conv_step_img.txt', 'wb') as fp:
    pickle.dump(netoutbatch, fp)

  base_net.load_state_dict(torch.load(Path1+r'\phase2_text'+str(i)+r'.pth', map_location=torch.device('cpu') ))
  netoutbatch=base_net.myforward(torch.FloatTensor(phit))
  with open(Path1+r"/"+'conv_step_text.txt', 'wb') as fp:
    pickle.dump(netoutbatch, fp)

########################################bulid train MLP##############################
def bulid_train_MLP_layer():
  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  with open(Path1+r"/"+'conv_step_img.txt', 'rb') as fp:
    input1 = pickle.load(fp)
  with open(Path1+r"/"+'conv_step_text.txt', 'rb') as fp:
    input2 = pickle.load(fp)

  input1 = torch.autograd.Variable(input1)
  input2 = torch.autograd.Variable(input2)



  input1=numpy.array(input1)
  input2=numpy.array(input2)
  inp=np.concatenate([input1,input2],1)
  input1=[]
  input2=[]
  hidden1=1500
  hidden2=980
  batch_size=200
  max_iterations=25000
  min_error=50
    


  target= datasets.Features172K().Get_phixtarget()
  
  model_mlp=NLR2(inp.shape[1],target.shape[1],hidden1,hidden2)
 
  torch.manual_seed(300)
  loss_fn = torch.nn.MSELoss()
  
 

  optimizer=torch.optim.SGD(model_mlp.parameters(), lr=0.002)
  epoch=max_iterations
  s=0
  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    
    for l in range(int(inp.shape[0]/batch_size)):
      
      item_batch = inp[l*batch_size+s:(l+1)*batch_size+s,:]
      target_batch=target[l*batch_size+s:(l+1)*batch_size+s,:]
      netoutbatch=model_mlp.myforward(torch.tensor(item_batch))
      loss = loss_fn(torch.tensor(target_batch),netoutbatch)
      losses.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
    if (total_loss<min_error):
      break
    print('iteration:',j,'MSE loss ',loss, 'total loss',total_loss)
    totallosses.append(total_loss)
    s+=1
    if s==48:
       s=0
    if (j%100==0) :
      torch.save(model_mlp.state_dict(), Path1+r'\mlp_3_2_net'+str(j)+r'.pth') 
      with open(Path1+r"/"+'loosses28112021.pkl', 'wb') as fp:
          pickle.dump( losses, fp)


  print('Finished Training')
  torch.save(model_mlp.state_dict(), Path1+r'\mlp_3_2_net_Final.pth') 

def test_Model_performance(file_index):
  with open(Path1+r"/"+'conv_step_img.txt', 'rb') as fp:
    input1 = pickle.load(fp)
  with open(Path1+r"/"+'conv_step_text.txt', 'rb') as fp:
    input2 = pickle.load(fp)

  input1 = torch.autograd.Variable(input1)
  input2 = torch.autograd.Variable(input2)
  input1=numpy.array(input1)
  input2=numpy.array(input2)
  inp=np.concatenate([input1,input2],1)
  hidden1=1400
  hidden2=980
  all_imgs= datasets.Features172K().Get_phix()
  

  model_mlp=NLR2(inp.shape[1],all_imgs.shape[1],hidden1,hidden2)
  model_mlp.load_state_dict(torch.load( Path1+r'\mlp_3_net'+str(file_index)+r'.pth', map_location=torch.device('cpu') ))
  all_queries=model_mlp.myforward(torch.FloatTensor(inp))
  all_queries=torch.autograd.Variable(all_queries)
  all_queries=numpy.array(all_queries)[:5000]

  inp=[]
  all_captions=datasets.Features172K().Get_all_captions()
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  with open (Path1+"/train_all_target_captions.txt", 'rb') as fp:
    all_target_captions = pickle.load(fp) 

  #all_target_captions=get_target_captions_train()
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  nn_result1=[]
  nn_result2=[]
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    sims1=input1[i:(i+1),:].dot(all_imgs.T)
    sims2=input2[i:(i+1),:].dot(all_imgs.T)

    nn_result.append(np.argsort(-sims[0, :])[:110])
    nn_result1.append(np.argsort(-sims1[0, :])[:110])
    nn_result2.append(np.argsort(-sims2[0, :])[:110])

    
  # compute recalls
  out = []
  out1=[]
  out2=[]
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  nn_result1 = [[all_captions[nn] for nn in nns1] for nns1 in nn_result1]
  nn_result2 = [[all_captions[nn] for nn in nns2] for nns2 in nn_result2]
  
  for k in [1, 5, 10, 50, 100]:
    
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out.append(str(k) + ' ---> '+ str(r*100))
    r = 0.0
    for i, nns1 in enumerate(nn_result1):
      if all_target_captions[i] in nns1[:k]:
        r += 1
    r /= len(nn_result1)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out1.append(str(k) + ' ---> '+ str(r*100))
    
    r = 0.0
    for i, nns2 in enumerate(nn_result2):
      if all_target_captions[i] in nns2[:k]:
        r += 1
    r /= len(nn_result2)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out2.append(str(k) + ' ---> '+ str(r*100))

    
  return out, out1, out2
def resume_train_MLP(file_index):
  
  with open(Path1+r"/"+'conv_step_img.txt', 'rb') as fp:
    input1 = pickle.load(fp)
  with open(Path1+r"/"+'conv_step_text.txt', 'rb') as fp:
    input2 = pickle.load(fp)

  input1 = torch.autograd.Variable(input1)
  input2 = torch.autograd.Variable(input2)
  input1=numpy.array(input1)
  input2=numpy.array(input2)
  inp=np.concatenate([input1,input2],1)
  hidden1=1400
  hidden2=980
  input1=[]
  input2=[]
  target= datasets.Features172K().Get_phixtarget()


  model_mlp=NLR2(inp.shape[1],target.shape[1],hidden1,hidden2)
  model_mlp.load_state_dict(torch.load( Path1+r'\mlp_3_net'+str(file_index)+r'.pth', map_location=torch.device('cpu') ))
  batch_size=16
  
  min_error=50
  loss_fn = torch.nn.MSELoss()
  
  losses=[]
  totallosses=[]
  optimizer=torch.optim.SGD(model_mlp.parameters(), lr=0.003)
  epoch=25000

  for j in range(epoch):
    total_loss=0
    for l in range(int(inp.shape[0]/batch_size)):
      
      item_batch = inp[l*batch_size:(l+1)*batch_size,:]
      target_batch=target[l*batch_size:(l+1)*batch_size,:]
      netoutbatch=model_mlp.myforward(torch.tensor(item_batch))
      loss = loss_fn(torch.tensor(target_batch),netoutbatch)
      losses.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
    if (total_loss<min_error):
      break
    print('iteration:',j,'MSE loss ',loss, 'total loss',total_loss)
    totallosses.append(total_loss)
    if (j%100==0) :
      torch.save(model_mlp.state_dict(), Path1+r'\mlp_3_net'+str(j+file_index+1)+r'.pth') 
      with open(Path1+r"/"+'loosses24112021resume300.pkl', 'wb') as fp:
          pickle.dump( losses, fp)


  print('Finished Training')
  torch.save(model_mlp.state_dict(), Path1+r'\mlp_3_net_Final.pth') 

def get_target_captions_train(count):
  trainset = datasets.Fashion200k(
        path=path2,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  all_target_captions=[]

  i=0
  for Data in tqdm(trainset):
    all_target_captions.append (Data['target_caption'])
    i+=1
    if (i>count):
      break
  return all_target_captions

def bulid_train_semantic_Hup():
  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  phit_caption=datasets.Features172K().Get_phit_image_caption()
  phix=datasets.Features172K().Get_phix()
  phix=Variable(torch.Tensor(phix))
  phit_caption=Variable(torch.Tensor(phit_caption))

  hidden1=1024
  hidden2=750
  batch_size=100
  max_iterations=25000
  min_error=15
    


  
  model_mlp=NLR2(phit_caption.shape[1],phix.shape[1],hidden1,hidden2)
 
  torch.manual_seed(300)
  loss_fn=torch.nn.CosineSimilarity()
  optimizer=torch.optim.SGD(model_mlp.parameters(), lr=0.002)
  epoch=max_iterations
  s=0
  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    
    for l in range(int(phix.shape[0]/batch_size)):
      
      item_batch = phit_caption[l*batch_size+s:(l+1)*batch_size+s,:]

      target_batch=phit_caption[l*batch_size+s:(l+1)*batch_size+s,:]
      netoutbatch=model_mlp.myforward(item_batch)
      loss = torch.mean(torch.abs(1-loss_fn(target_batch,netoutbatch)))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(loss)
      optimizer.step()
      total_loss+=loss
    if (total_loss<min_error):
      break
    print('iteration:',j,'COS similarity loss ',loss, 'total loss',total_loss)
    totallosses.append(total_loss)
    s+=1
    if s==48:
       s=0
    if (j%100==0) :
      torch.save(model_mlp.state_dict(), Path1+r'\mlp_semantichup_net'+str(j)+r'.pth') 
      with open(Path1+r"/"+'loossessemantichup.pkl', 'wb') as fp:
          pickle.dump( losses, fp)


  print('Finished Training')
  torch.save(model_mlp.state_dict(), Path1+r'\mlp_semantichup_net_Final.pth') 
def save_semantic_hup_output():
  phit_query_caption=datasets.Features172K().Get_phit()
  phix=datasets.Features172K().Get_phix()
  phix=Variable(torch.Tensor(phix))
  phit_query_caption=Variable(torch.Tensor(phit_query_caption))

  hidden1=1024
  hidden2=750
  model_mlp=NLR2(phit_query_caption.shape[2],phix.shape[1],hidden1,hidden2)
  model_mlp.load_state_dict(torch.load( Path1+r'\mlp_semantichup_net_Final.pth', map_location=torch.device('cpu') ))
  semantic_query_caption=model_mlp.myforward(phit_query_caption)
  with open(Path1+r'\semantic_query_caption.txt', 'wb') as fp:
    pickle.dump(semantic_query_caption, fp)
  input1 = torch.autograd.Variable(semantic_query_caption)
  input1=torch.squeeze(input1)
  input2 = torch.autograd.Variable(phix)
  input1=numpy.array(input1)
  input2=numpy.array(input2)
  inp=np.concatenate([input1,input2],1)

  with open(Path1+r'\squery_with_phix.txt', 'wb') as fp:
    pickle.dump(inp, fp)


def bulid_train_final_net_with_semantic_Hup():
  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  inp=datasets.Features172K().Get_squery_caption_with_phix()
  target=datasets.Features172K().Get_phixtarget()
  inp=Variable(torch.Tensor(inp))
  target=Variable(torch.Tensor(target))

  hidden=900
  batch_size=500
  max_iterations=25000
  min_error=12
  model_mlp=NLR3(inp.shape[1],target.shape[1],hidden)
 
  torch.manual_seed(30)
  loss_fn=torch.nn.CosineSimilarity()
  optimizer=torch.optim.SGD(model_mlp.parameters(), lr=0.002)
  epoch=max_iterations
  s=0
  sweep_range=inp.shape[0]%batch_size

  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    
    for l in range(int(inp.shape[0]/batch_size)):
      
      item_batch = inp[l*batch_size+s:(l+1)*batch_size+s,:]

      target_batch=target[l*batch_size+s:(l+1)*batch_size+s,:]
      netoutbatch=model_mlp.myforward(item_batch)
      loss = torch.mean(torch.abs(1-loss_fn(target_batch,netoutbatch)))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(loss)
      optimizer.step()
      total_loss+=loss
    if (total_loss<min_error):
      break
    print('iteration:',j,'COS similarity loss ',loss, 'total loss',total_loss)
    totallosses.append(total_loss)
    s+=1
    if s==sweep_range:
       s=0
    if (j%500==0) :
      torch.save(model_mlp.state_dict(), Path1+r'\3final_net_with_Shup'+str(j)+r'.pth') 
      with open(Path1+r"/"+'3lossesfinal_net_with_Shup.pkl', 'wb') as fp:
          pickle.dump( losses, fp)


  print('Finished Training')
  torch.save(model_mlp.state_dict(), Path1+r'\3final_net_with_Shup_Final.pth') 


def resume_train_final_net_with_semantic_Hup(start_no):
  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  inp=datasets.Features172K().Get_squery_caption_with_phix()
  target=datasets.Features172K().Get_phixtarget()
  inp=Variable(torch.Tensor(inp))
  target=Variable(torch.Tensor(target))

  hidden=900
  batch_size=500
  max_iterations=25000
  min_error=12
  model_mlp=NLR3(inp.shape[1],target.shape[1],hidden)
  model_mlp.load_state_dict(torch.load( Path1+r'\3final_net_with_Shup'+str(start_no)+r'.pth', map_location=torch.device('cpu') ))
 
  loss_fn=torch.nn.CosineSimilarity()
  optimizer=torch.optim.SGD(model_mlp.parameters(), lr=0.001)
  epoch=max_iterations

  s=0
  sweep_range=inp.shape[0]%batch_size
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    
    for l in range(int(inp.shape[0]/batch_size)):
      
      item_batch = inp[l*batch_size+s:(l+1)*batch_size+s,:]

      target_batch=target[l*batch_size+s:(l+1)*batch_size+s,:]
      netoutbatch=model_mlp.myforward(item_batch)
      loss = torch.mean(torch.abs(1-loss_fn(target_batch,netoutbatch)))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #losses.append(loss)
      optimizer.step()
      total_loss+=loss
    if (total_loss<min_error):
      break
    print('iteration:',j,'COS similarity loss ',loss, 'total loss',total_loss)
    totallosses.append(total_loss)
    s+=1
    if s==sweep_range:
       s=0
    if (j%200==0) :
      torch.save(model_mlp.state_dict(), Path1+r'\3final_net_with_Shup'+str(j+start_no)+r'.pth') 
      with open(Path1+r"/"+'3lossesfinal_net_with_Shup'+str(start_no)+'.pkl', 'wb') as fp:
          pickle.dump( totallosses, fp)


  print('Finished Training')
  torch.save(model_mlp.state_dict(), Path1+r'\3final_net_with_Shup_Final.pth') 

def semantic_Model_performance(file_no):
  inp=datasets.Features172K().Get_squery_caption_with_phix()
  target=datasets.Features172K().Get_phixtarget()
  inp=Variable(torch.Tensor(inp))
  all_captions=datasets.Features172K().Get_all_captions()
  all_target_captions=datasets.Features172K().Get_all_target_captions()

  hidden1=2024
  hidden2=1024
  model_mlp=NLR2(inp.shape[1],target.shape[1],hidden1,hidden2)
  model_mlp.load_state_dict(torch.load( Path1+r'\final_net_with_Shup'+str(file_no)+r'.pth', map_location=torch.device('cpu') ))

  netout=model_mlp.myforward(inp)
  netout = torch.autograd.Variable(netout)
  #target = torch.autograd.Variable(target)

  netout=numpy.array(netout)
  #target=numpy.array(target)

  for i in range(netout.shape[0]):
    netout[i,:]/=np.linalg.norm(netout[i, :])
  for i in range(target.shape[0]):
    target[i,:]/=np.linalg.norm(target[i, :])
  
  print('test normalization', np.linalg.norm(target[1, :]),np.linalg.norm(netout[1, :]))
  # match test queries to target images, get nearest neighbors
  nn_result = []
  
  for i in tqdm(range(int(netout.shape[0]/20))):
    sims = netout[i:(i+1), :].dot(target.T)
    
    nn_result.append(np.argsort(-sims[0, :])[:110])
  
    
  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  
  
  for k in [1, 5, 10, 50, 100]:
    
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out.append(str(k) + ' ---> '+ str(r*100))
    r = 0.0

    
  return out 
def regression_study(st):
  # with open (Path1+"\\BetatrainLoaded.txt", 'rb') as fp:
  #   Beta = pickle.load(fp) 

  inp1=datasets.Features172K().Get_phix()   #[:40000,:] 
  inp2=datasets.Features172K().Get_phit()  #[:40000,:] 
  inp2=np.concatenate(inp2)
  #inp2=inp2[:40000,:]
  target=datasets.Features172K().Get_phixtarget()  #[:40000,:]  #[:40000,:]
  with open(Path1+r"/"+'target_captions_all_captions.pckl', 'rb') as fp:
      captions_target_list=pickle.load( fp)  
  reg1 = LinearRegression().fit(inp1, target)
  reg2 = LinearRegression().fit(inp2, target)
  newinp1=reg1.predict(inp1)
  newinp2=reg2.predict(inp2)
  newinp=np.concatenate((newinp1,newinp2),axis=1)
  reg3 = LinearRegression().fit(newinp, target)
  target_sout=reg3.predict(newinp)
  cnt=np.zeros(5)
  rng=[1,5,10,50,100]
  rng=np.array(rng)
  for i in range(np.shape(target_sout)[0]):
    tlist=argsort((np.square(inp1-target[i,:])).sum(1))[:100]
    if (i%200==0):
      print(set(captions_target_list[i]).intersection(set(tlist)))
    for j in range(5):
      if (set(captions_target_list[i]).intersection(set(tlist[:j]))) !=set():
       cnt[j] +=1
    if (i%200 ==0):
      print('counts',cnt, 'percent',100*cnt/(i+1), 'index = ',i+1)
    
  print('precent= ',rng ,' are of values ',100*cnt/np.shape(target_sout)[0])
def datasets_check():
    print('Querys Imgs Lenght 172k:',len(datasets.Feature172KOrg().PhixQueryImg))
    print('Querys Captions Lenght 172k:',len(datasets.Feature172KOrg().PhitQueryCaption))
    print('Querys Modifier Text Lenght 172k:',len(datasets.Feature172KOrg().PhitQueryMod))
    all_target_captions=datasets.Feature172KOrg().PhixTargetImg
    print('Target Imgs Lenght 172k:',len(datasets.Feature172KOrg().PhixTargetImg))
    print('Target Captions Lenght 172k:',len(datasets.Feature172KOrg().PhitTargetCaption))
 
    ########## 33K #########

    print('Querys Imgs Lenght 33K:',len(datasets.Features33KOrg().PhixQueryImg))
    print('Querys Captions Lenght 33K:',len(datasets.Features33KOrg().PhitQueryCaption))
    print('Querys Modifier Text Lenght 33K:',len(datasets.Features33KOrg().PhitQueryMod))
    print('Target Imgs Lenght 33K:',len(datasets.Features33KOrg().PhixTargetImg))
    print('Target Captions Lenght 33K:',len(datasets.Features33KOrg().PhitTargetCaption))
    print('All Imgs Unique Lenght 29K:',len(datasets.Features33KOrg().PhixAllImages))
    print('All Imgs Captions Unique Lenght 29K:',len(datasets.Features33KOrg().PhitAllImagesCaptions))

def save_captions_values():
 
  train = datasets.Fashion200k(
        path=path2,
        split='train',
        transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize(224),
          torchvision.transforms.CenterCrop(224),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
       ]))
  all_target_captions =[]
  all_cap=datasets.Features172K().Get_all_captions()
  all_tar_cap=datasets.Features172K().Get_all_target_captions()
  train.caption_index_init_()
  print(len(train))
  for t in(train):
    all_target_captions += [t['target_caption']]
  with open(Path1+r"/"+'Features172Kall_target_captions.pckl', 'wb') as fp:
    pickle.dump(all_target_captions, fp)

  return 
  test_dataset = datasets.Fashion200k(
        path=path2,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))


  all_img_captions = []
  all_target_captions = []
  all_mods=[]
  length=len(test_dataset)
  test_queries = test_dataset.get_test_queries()
  for t in tqdm(test_queries):
    all_mods += [t['mod']['str']]
    all_img_captions += [t['source_caption']]
    all_target_captions += [t['target_caption']]
    
    
  with open(Path1+r"/"+'Features33Kall_image_caption.pckl', 'wb') as fp:
    pickle.dump(all_img_captions, fp)

  with open(Path1+r"/"+'Features33Kall_target_captions.pckl', 'wb') as fp:
    pickle.dump(all_target_captions, fp)

  with open(Path1+r"/"+'Features33Kall_mods.pckl', 'wb') as fp:
    pickle.dump(all_mods, fp)
  

  print('172 Finished')
  
def prepare_dataset():
    with open(Path1+r"/"+'Features33Kall_image_caption.pckl', 'rb') as fp:
      all_img_captions_test=pickle.load( fp)

    with open(Path1+r"/"+'Features33Kall_target_captions.pckl', 'rb') as fp:
      all_target_captions_test=pickle.load(fp)

    #with open(Path1+r"/"+'Features33Kall_mods.pckl', 'rb') as fp:
    #  all_mods_test=pickle.load( fp)
    print(" data test done ")
    captions_target_list=[]
    for i in range(len(all_target_captions_test)):
      itemlist=[j for j,x in enumerate(all_img_captions_test) if x==all_target_captions_test[i]]

      captions_target_list.append(itemlist)
    with open(Path1+r"/"+'target_captions_all_captions33k.pckl', 'wb') as fp:
      pickle.dump(captions_target_list, fp)
  
    with open(Path1+r"/"+'Features172Kall_image_caption.pckl', 'rb') as fp:
      all_img_captions_train=pickle.load( fp)

    with open(Path1+r"/"+'Features172Kall_target_captions.pckl', 'rb') as fp:
      all_target_captions_train=pickle.load(fp)

      

    captions_target_list=[]
  
    for i in range(len(all_target_captions_train)):
      itemlist=[j for j,x in enumerate(all_img_captions_train) if x==all_target_captions_train[i]]
      if (i%500==0):
        print(" train 172",i)
      captions_target_list.append(itemlist)
    with open(Path1+r"/"+'target_captions_all_captions172k.pckl', 'wb') as fp:
      pickle.dump(captions_target_list, fp)

def print_element(no):
  
    train = datasets.Fashion200k(
      path=path2,
      split='train',
      transform=torchvision.transforms.Compose([
      torchvision.transforms.Resize(224),
      torchvision.transforms.CenterCrop(224),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
      ]))
    im=train[no]
    target_id =im['target_img_id']
    
    combined_text='img:'+im['source_caption']+' Target:' + im['target_caption']+' mod:' + im['mod']['str']
    print('element =',no,'target id ',target_id)
    print(combined_text)

def inspect_case():
  with open(Path1+r"/"+'Features172Kall_image_caption.pckl', 'rb') as fp:
    all_img_captions_train=pickle.load( fp)

  with open(Path1+r"/"+'Features172Kall_target_captions.pckl', 'rb') as fp:
    all_target_captions_train=pickle.load(fp)
  print(all_img_captions_train[:10])
  print('separator')
  print(all_target_captions_train[:10])
  print_element(0)
  print_element(50377)
  print_element(118871)
    
if __name__ == '__main__': 
    
  #phase2_network()  img_model_file,text_model_file,flag  
  #asbook1, model=Phase2_test_models_get_orignal("4iCovLinearflr006w12.pth","4Dtlr006w124000.pth",3)
  #name="joint"
  #print(name,' L oaded As PaPer: ',asbook1, '\n  model generated      ',model, '\n  ' )
  #datasets_check()
  #prepare_dataset()
  ###########phase2_two_network_model_train()
  #############select_best_saved_model()
  #for i in range(5):

    #print(semantic_Model_performance(i*100))
  #resume_train_final_net_with_semantic_Hup(4500)
  #inspect_case()
  #regression_study(1)
  save_captions_values()
  
  #bulid_train_final_net_with_semantic_Hup()
  #save_semantic_hup_output()
  #bulid_train_semantic_Hup()

  #bulid_train_MLP_layer()
  #resume_train_MLP(300)
  #print('results :' , test_Model_performance(401))
  #all_target_captions=get_target_captions_train(55000)
  #with open(Path1+r"/"+'train_all_target_captions.txt', 'wb') as fp:
  #  pickle.dump(all_target_captions, fp)

  
  


  #model_file='4Dilr006w124000.pth'
  #k=0
  #path2=path2=r"C:\MMaster\Files"
  
  #dataset_used = datasets.Fashion200k(
  #      path=path2,
  #      path=path2,
  #      split='train',
  #      transform=torchvision.transforms.Compose([
  #          torchvision.transforms.Resize(224),
  #          torchvision.transforms.CenterCrop(224),
  #          torchvision.transforms.ToTensor(),
  #          torchvision.transforms.Normalize([0.485, 0.456, 0.406],
  #                                           [0.229, 0.224, 0.225])
  #      ]))

  #for i in range(0,170000,500):
  #  k+= CBIR(i,1, 500,model_file,dataset_used)

  #print('total found is ',k)

  
    





    