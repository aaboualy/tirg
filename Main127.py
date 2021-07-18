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
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'

  datasets.Features172K().SavetoFilesImageSource(Path1+r'/dataset172', trig, train,opt)
  print('172 Finished')

#################  Beta Regression (Train Set after Trig && Tested on Test Set)  Section   #################

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

    asbook = test_retrieval.testLoaded(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)


#################  Beta From Test Set Section   #################

     

def getbetatestLoaded():
    
  imgdata = datasets.Features33K().Get_all_images()
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

  

  with open(Path1+r"/"+'BetatestLoaded.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestestloaded():
  
  with open (Path1+"\\BetatestLoaded.txt", 'rb') as fp:
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
  with open (Path1+"\\17720\\Cos\\loosses2.pkl", 'rb') as fp:
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
  plt.title('Cos Loss Graph')
  plt.show()



if __name__ == '__main__': 
    getlossesMeanSquare()
    getlossesCos()

    