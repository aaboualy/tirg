import torch
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




Path1=r"D:\personal\master\MyCode\files"

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
  trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'

  #datasets.Features172K().SavetoFilesImageSource(Path1+r'/dataset172', trig, train,opt)
  #datasets.Features33K().SavetoFiles2(Path1+r'/dataset33', trig, test,opt)
  #datasets.Features33K().SavetoFiles3(Path1+r'/dataset33', trig, test,opt)

  datasets.Features172K().SavetoFilesold(Path1+r'/dataset172', trig, train,opt)
  datasets.Features33K().SavetoFilesold(Path1+r'/dataset33', trig, test,opt)
  
  #print('172 Finished')
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
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.Sigmoid(),torch.nn.Linear(nethidden1, nethidden2),torch.nn.Linear(nethidden2, netout))
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
  imgdata = datasets.Features172K().Get_all_images()#[:139999]
  all_queries1 = datasets.Features172K().Get_all_queries()#[:139999]

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

  

 

  






######################  RUN  ##########################

# train  As PaPer:  ['1 ---> 33.21', '5 ---> 60.67', '10 ---> 73.18', '50 ---> 92.97999999999999', '100 ---> 97.19']
# test  As PaPer:  ['1 ---> 14.04121863799283', '5 ---> 34.268219832735966', '10 ---> 42.41338112305854', '50 ---> 65.31660692951016', '100 ---> 73.66487455197132']


if __name__ == '__main__': 
    
    #getbetatrainLoadedold()
    GetValuesRandomForestRegressor()

    





    