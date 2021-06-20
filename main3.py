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
Path1=r"C:\MMaster\Files"



Path1=r"D:\personal\master\MyCode\files"
#Path1=r"C:\MMaster\Files"


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



#################  Beta From Test Set Section   #################

def getbeta():
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

  


  with open(Path1+r"/"+'testBetaNormalizedG.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValues():
  
  with open (Path1+"/testBetaNormalized.txt", 'rb') as fp:
    Nbeta = pickle.load(fp) 

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
  
  for name, dataset in [ ('train', train),('test', test)]: #('train', trainset),
     
     betaNor = test_retrieval.testWbeta(opt, trig, dataset,Nbeta)
     print(name,' BetaNormalized: ',betaNor)

     asbook = test_retrieval.test(opt, trig, dataset)
     print(name,' As PaPer: ',asbook)
    
     
#################  Beta From Train Set Section   #################

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
 
  #m = nn.ReLU()
  

  for i in range(172048): #172048
    print('get images=',i,end='\r')
    item = train[i]
    imgs += [item['source_img_data']]
    mods += [item['mod']['str']]
    target += [item['target_img_data']]
    

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

  

  Ntrig2=[]

  
 
  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  for i in range(trigdata.shape[0]):
    Ntrig2.append(np.insert(trigdata[i],0, 1))

  print("Ntrig2 shape %d  first elemnt %d",Ntrig2[0] )
  Ntrig2=np.array(Ntrig2)
  Ntrigdata1=Ntrig2.transpose()
  X1=np.matmul(Ntrigdata1,Ntrig2)  
  X2=np.linalg.inv(X1)
  X3=np.matmul(X2,Ntrigdata1)  
  Nbeta=np.matmul(X3,imgdata) 

  

  with open(Path1+r"/"+'Betatrain.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuestrain():
  
  with open (Path1+"\\Betatrain.txt", 'rb') as fp:
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
    print(name,' BetaNormalized: ',betaNor)

    # asbook = test_retrieval.test(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)


#################  Get Average Beta   #################

def GetAverageBeta():
  with open (Path1+"/Beta.txt", 'rb') as fp:
    BetaTrain = pickle.load(fp) 

  with open (Path1+"/testBetaNormalized.txt", 'rb') as fp:
    BetaTest = pickle.load(fp) 

  BetaAvg1= np.add(BetaTrain, BetaTest)
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

    asbook = test_retrieval.test(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)

#################  Beta From Train & Test Set Section   #################

def getbetaall():
  

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
    
    
    imgs += [train.get_img(Data['source_img_id'])]
    mods += [Data['mod']['str']]
    target +=[train.get_img(Data['target_img_id'])]

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

  

  with open(Path1+r"/"+'Betaall.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

def GetValuesall():
  
  with open (Path1+"/Betaall.txt", 'rb') as fp:
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
  
  for name, dataset in [ ('train', trainset)]: #('train', trainset), ,('test', testset)
    
    betaNor = test_retrieval.testWbeta(opt, trig, dataset,BetaNormalize)
    print(name,' BetaNormalized: ',betaNor)

    # asbook = test_retrieval.test(opt, trig, dataset)
    # print(name,' As PaPer: ',asbook)

def getvaluespdf():
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
 
  #m = nn.ReLU()
  

  for i in range(172048): #172048
    print('get images=',i,end='\r')
    item = train[i]
    imgs += [item['source_img_data']]
    mods += [item['mod']['str']]
    target += [item['target_img_data']]
    

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
    
    target=[]

  for i in range(trigdata.shape[0]):
    trigdata[i, :] /= np.linalg.norm(trigdata[i, :])
  
  for i in range(imgdata.shape[0]):
    imgdata[i, :] /= np.linalg.norm(imgdata[i, :])


  print(trigdata)
  print(imgdata)
  with open(Path1+r"/"+'traindata.txt', 'wb') as fp:
    pickle.dump(trigdata, fp) 

  with open(Path1+r"/"+'imgdata.txt', 'wb') as fp:
    pickle.dump(imgdata, fp)

class NLR(nn.Module):
  def __init__(self,insize,outsize,hidden):
    super().__init__()
    self.nlmodel= torch.nn.Sequential(torch.nn.Linear(insize, hidden),torch.nn.Sigmoid(),torch.nn.Linear(hidden, outsize))
  def myforward (self,x11):
    p=self.nlmodel(x11)
    return p


def getNLP():
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

  dtsz, indm, hddm, oudm = 172048, 513, 700, 512

  loss_fn = torch.nn.MSELoss(reduction='sum')
  torch.manual_seed(3)
  model=NLR(indm,oudm,hddm)
  #model=model.cuda()
  torch.manual_seed(3)

  criterion=nn.MSELoss()
  optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
  epoch=50

  losses=[]
  
  for j in range(epoch):
    for l in range(dtsz): #172048
      print('Epoch:',j,' get images=',l,end='\r')
      item = train[l]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      target += [item['target_img_data']]
      
      imgs = torch.stack(imgs).float()
      imgs = torch.autograd.Variable(imgs)#.cuda()
      
      target = torch.stack(target).float()
      target = torch.autograd.Variable(target)#.cuda()

      f = trig.compose_img_text(imgs, mods).data.cpu().numpy()
      f2 = trig.extract_img_feature(target).data.cpu().numpy()

      for i in range(f.shape[0]):
        f[i, :] /= np.linalg.norm(f[i, :])
      
      for i in range(f2.shape[0]):
        f2[i, :] /= np.linalg.norm(f2[i, :]) 

      for i in range(f.shape[0]):
        trigdata =np.insert(f[i],0, 1)

      trigdata=torch.from_numpy(trigdata)
      f2=torch.from_numpy(f2)

      yp=model.myforward(trigdata)
      loss=criterion(yp,f2)
      if(l%20000 == 0):
        print("epoch ",j, "loss ", loss.item())
      losses.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      imgs = []
      mods = []
      trigdata=[]
      target=[]
      imgdata=[]




  print('Finished Training')
  torch.save(model.state_dict(), Path1+r'\NLP2.pth') 
  
def resultsNLP():
  
  
  dtsz, indm, hddm, oudm = 172048, 513, 700, 512


  model=NLR(indm,oudm,hddm)
  model.load_state_dict(torch.load(Path1+r'\NLP.pth' , map_location=torch.device('cpu') ))
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
    
    NLP = test_retrieval.testNLP(opt, trig, dataset,model)
    print(name,' NLP: ',NLP)

    asbook = test_retrieval.test(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)

def savevaluestofile():
    
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
  alldata=[]
  #m = nn.ReLU()
  

  for i in range(172048): #172048
    print('get images=',i,end='\r')
    item = train[i]
    imgs += [item['source_img_data']]
    mods += [item['mod']['str']]
    target += [item['target_img_data']]
    

    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()

    target = torch.stack(target).float()
    target = torch.autograd.Variable(target)
    
    f2 = trig.extract_img_feature(target).data.cpu().numpy() 
    # trigdata.append(f[0])
    # imgdata.append(f2[0])

    opsig={
                  'SourceTrig':f[0],
                  'TargetData':f2[0],
                  'IDX':i
    }
    alldata.append(opsig)
    imgs = []
    mods = []
    trigdata=[]
    target=[]
    imgdata=[]




  with open(Path1+r"/"+'TrigImgData172.txt', 'wb') as fp:
    pickle.dump(alldata, fp)


def Savevaluestest():
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
  alldata=[]
  
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

    
    opsig={
                  'SourceTrig':f[0],
                  'TargetData':f2[0],
                  'IDX':Data['source_img_id']
    }
    alldata.append(opsig)
    all_captions = [img['captions'][0] for img in test.imgs]

    imgs = []
    mods = []
    trigdata=[]
    target=[]
    imgdata=[]

  with open(Path1+r"/"+'allcaptions.txt', 'wb') as fp:
    pickle.dump(all_captions, fp)



  with open(Path1+r"/"+'TrigImgDatatestset.txt', 'wb') as fp:
    pickle.dump(alldata, fp)
    
def trainsaveddataresultsa():
  with open (Path1+"\\TrigImgData172.txt", 'rb') as fp:
    Datasaved172 = pickle.load(fp) 

  with open (Path1+"\\TrigImgDatatestset.txt", 'rb') as fp:
    Datasavedtest = pickle.load(fp) 
  
  with open (Path1+"\\Betatrain.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 

 
    
    #betaNor = test_retrieval.testWbetaWsaveddataa(BetaNormalize,Datasaved172)
    #print('trained',' BetaNormalized: ',betaNor)
    betaNor = test_retrieval.testWbetaWsaveddataa(BetaNormalize,Datasavedtest)
    print('test',' BetaNormalized: ',betaNor)

def trainsaveddataresults():
  with open (Path1+"\\TrigImgData172.txt", 'rb') as fp:
    Datasaved172 = pickle.load(fp) 

  with open (Path1+"\\TrigImgDatatestset.txt", 'rb') as fp:
    Datasavedtest = pickle.load(fp) 
  
  with open (Path1+"\\Betatrain.txt", 'rb') as fp:
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
    
    betaNor = test_retrieval.testWbetaWsaveddata(opt, trig, dataset,BetaNormalize,Datasaved172,Datasavedtest)
    print(name,' BetaNormalized: ',betaNor)

def Save_GetValues():
  
  
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
  
  #for name, dataset in [ ('test', test),('train', train)]: #('train', trainset),
  for name, dataset in [ ('test', test)]: #('train', trainset),
          
     asbook = test_retrieval.test_and_save(opt, trig, dataset)
     print(name,' As PaPer: ',asbook)

def print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld):
  print(' Experiment setup : ', file = sourceFile)
  if (test_train==1):
    print('Dataset:Training Data set', file = sourceFile)
  else:
    print('Dataset:Testing Data set', file = sourceFile)

  if (normal_beta==0):
    print(' Trig', file = sourceFile)
  else:
    print(' Trig followed by Regression network', file = sourceFile)
  if (normal_beta==1):
    if (create_load==0):
      print(' Regression Network Created, save to file', file = sourceFile)
    else:
      print(' Regression Network Loaded from file ', file = sourceFile)
    print(' = ',filename, file = sourceFile)
    if (normal_normalize==0):
      print(' Regression done without normalization ', file = sourceFile)
    else:
      print(' Regression done on normalized vectors ', file = sourceFile)
  else:
    print('       ', file=sourceFile)
  if (dot_eucld==0):
    print(' Distance: Cos Angle between vectors ', file = sourceFile)
  else:
    print(' Distance: Eucledian  ', file = sourceFile)
  print(' Dataset size Divider ', set_size_divider, file = sourceFile)
  print(' Experiment Outcome: - ','\n',out,'\n', file = sourceFile)

def results():
  sourceFile = open(Path1+r"/"+'results'+time.strftime("%Y%m%d-%H%M%S")+'.txt', 'w')
  test_train=0
  normal_beta=0
  set_size_divider=1
  normal_normalize=0
  create_load=0
  filename='na'
  dot_eucld=0
  # 1
  print(' 1', file=sourceFile)
  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=17.2
  # 2
  print(' 2', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  normal_beta=1
  create_load=0
  filename='REGTR10ND.BTA'
  # 3
  print(' 3', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10ND.BTA'
  # 4
  print(' 4', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTS33ND.BTA'
  # 5
  print(' 5', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  test_train=1
  set_size_divider=1
  normal_beta=0
  create_load=0
  filename='na'
  # 6
  print(' 6', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)

  test_train=1
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTR172ND,BTA'
  # 7
  print(' 7', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
 
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR172ND,BTA'
  # 8
  print(' 8', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  ###################NNORMALIZED BETA##############################################################
  test_train=3
  normal_beta=1
  set_size_divider=17.2
  normal_normalize=0
  create_load=0
  filename='REGTR10NND.BTA'
  dot_eucld=0
  test_train=1
  # 3NN
  print(' 3NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10NND.BTA'
  # 4 NN
  print(' 4NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTR172NND,BTA'
  # 7 NN
  print(' 7NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
 
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR172NND,BTA'
  # 8 NN
  print(' 8NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  ###################eucledian##############################################################
  test_train=0
  normal_beta=0
  set_size_divider=1
  normal_normalize=0
  create_load=0
  filename='na'
  dot_eucld=1
  # 1 E
  print(' 1 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=17.2
  # 2 E
  print(' 2 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  normal_beta=1
  create_load=0
  filename='REGTR10NE.BTA'
  # 3 E
  print(' 3 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10NE.BTA'
  # 4 E
  print(' 4 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTS33NE.BTA'
  # 5 E
  print(' 5 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  test_train=1
  set_size_divider=1
  normal_beta=0
  create_load=0
  filename='na'
  # 6 E
  print(' 6 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)

  sourceFile.close()

# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates the retrieval model."""
import numpy as np
import pickle
import torch
from tqdm import tqdm as tqdm
from scipy.spatial import distance



def test(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)#.cuda()
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)#.cuda()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      print('get images=',i,end='\r')
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) >= opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.compose_img_text(imgs, mods).data.cpu().numpy() #.cuda()
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) >= opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0).data.cpu().numpy() #.cuda()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
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

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out

def testWbeta(opt, model, testset,beta):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()
 

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        for j in range(len(f)): 
          # for i in range(f.shape[0]):
          #   f[i, :] /= np.linalg.norm(f[i, :])
          f[j, :] /= np.linalg.norm(f[j, :])

          X1 = np.insert(f[j],0, 1)
          X2=np.matmul(X1,beta) 
          f[j]=X2
        
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      print('get images=',i,end='\r')
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) >= opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        for j in range(len(f)): 
          #for i in range(f.shape[0]):
            #f[i, :] /= np.linalg.norm(f[i, :])
          f[j, :] /= np.linalg.norm(f[j, :])
          X1 = np.insert(f[j],0, 1)
          X2=np.matmul(X1,beta) 
          f[j]=X2
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) >= opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
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

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out

def testNLP(opt, model, testset,model2):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()
 

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        for i in range(f.shape[0]):
          f[i, :] /= np.linalg.norm(f[i, :])
        f =np.insert(f,0, 1)
        f=np.expand_dims(f, axis=0)
        f=torch.from_numpy(f)
        
        f=model2.myforward(f).data.cpu().numpy()


        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(10000):
      print('get images=',i,end='\r')
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) >= opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        for i in range(f.shape[0]):
          f[i, :] /= np.linalg.norm(f[i, :])
        f =np.insert(f,0, 1)
        f=np.expand_dims(f, axis=0)
        f=torch.from_numpy(f)
        
        f=model2.myforward(f).data.cpu().numpy()


        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) >= opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  # for i in range(all_queries.shape[0]):
  #   all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
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

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out

def testWbetaWsaveddata(opt, model, testset,beta,savedtrain,savedtest):
  """Tests a model over the given testset."""
  model.eval()
  
  test_queries = testset.get_test_queries()
 

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in range(len(savedtest)):
      print('get testdata=',t,end='\r')
      f=savedtest[t]['SourceTrig']
      f=np.expand_dims(f, axis=0)
      for j in range(len(f)): 
        
        f[j, :] /= np.linalg.norm(f[j, :])
        X1 = np.insert(f[j],0, 1)
        X2=np.matmul(X1,beta) 
        f[j]=X2
      
      all_queries += [f]
      imgs = []
      mods = []

    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    
    
    for i in range(10000):
      print('get images=',i,end='\r')
      item = testset[i]
      f=savedtrain[i]['SourceTrig']
      f=np.expand_dims(f, axis=0)
      for j in range(len(f)): 
          
        f[j, :] /= np.linalg.norm(f[j, :])
        X1 = np.insert(f[j],0, 1)
        X2=np.matmul(X1,beta) 
        f[j]=X2
        
      all_queries += [f]
      imgs = []
      mods = []
      imgs0 += [savedtrain[i]['TargetData']]
      all_imgs += [imgs0]
      imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
      f=[]
    
    
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
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

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out

def test_and_save(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  all_captions=[]
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      all_captions += [t['source_caption']]
      all_target_captions += [t['target_caption']]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)#.cuda()
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    #all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)#.cuda()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    for i in range(len(testset)):
      
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) >= opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.compose_img_text(imgs, mods).data.cpu().numpy() #.cuda()
        all_queries += [f]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) >= opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0).data.cpu().numpy() #.cuda()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['source_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  if test_queries:
   with open(Path1+r"/"+'test_test_queries.pkl', 'wb') as fp:
    pickle.dump(test_queries, fp)
   with open(Path1+r"/"+'test_all_queries.pkl', 'wb') as fp:
    pickle.dump(all_queries, fp)

   with open(Path1+r"/"+'test_all_imgs.pkl', 'wb') as fp:
    pickle.dump(all_imgs, fp)
   with open(Path1+r"/"+'test_all_captions.pkl', 'wb') as fp:
    pickle.dump(all_captions, fp)
   with open(Path1+r"/"+'test_all_target_captions.pkl', 'wb') as fp:
    pickle.dump(all_target_captions, fp)
  else:
   with open(Path1+r"/"+'test_queries172k.pkl', 'wb') as fp:
    pickle.dump(test_queries, fp)

   with open(Path1+r"/"+'all_queries172k.pkl', 'wb') as fp:
    pickle.dump(all_queries, fp)
   with open(Path1+r"/"+'all_imgs172k.pkl', 'wb') as fp:
    pickle.dump(all_imgs, fp)
   with open(Path1+r"/"+'all_captions172k.pkl', 'wb') as fp:
    pickle.dump(all_captions, fp)
   with open(Path1+r"/"+'all_target_captions172k.pkl', 'wb') as fp:
    pickle.dump(all_target_captions, fp)




  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    if test_queries:
      sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
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

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out
def test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize,sz,dot_eucld):
  # test_queries:
  if test_train==0:
   with open(Path1+r"/"+'test_test_queries.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'test_all_queriesG.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'test_all_imgsG.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'rb') as fp:
    all_captions=pickle.load( fp)
   with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  else:
   with open(Path1+r"/"+'test_queries172k.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'all_queries172k.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'all_imgs172k.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   with open(Path1+r"/"+'all_captions172k.pkl', 'rb') as fp:
    all_captions=pickle.load( fp)
   with open(Path1+r"/"+'all_target_captions172k.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  if (normal_beta==1 ):
    if(create_load==0):
    #################################
      new_all_queries=np.zeros((all_queries.shape[0],all_queries.shape[1]+1))
      for i in range(all_queries.shape[0]):
        f=all_queries[i,:]
        if (normal_normalize==1):
          f/=np.linalg.norm(f)
        f=np.insert(f,0,1)
        new_all_queries[i,:]=f
      if (normal_normalize==1):
        for i in range(all_imgs.shape[0]):
          all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
      
      new_all_queriest=new_all_queries.transpose()
      X1=np.matmul(new_all_queriest,new_all_queries)  
      X2=np.linalg.inv(X1)
      X3=np.matmul(X2,new_all_queriest)  
      beta=np.matmul(X3,all_imgs) 
      new_all_queries=[]
      new_all_queriest=[]
     #################################
      with open(Path1+r"/"+filename, 'wb') as fp:
        pickle.dump( beta, fp)
    else:
      with open(Path1+r"/"+filename, 'rb') as fp:
        beta=pickle.load( fp)
    for t in range(int(len(all_queries)/sz)):
      if (t%100==0):
        print('get testdata=',t,end='\r')
      f=all_queries[t,:]
      if (normal_normalize==1):
        f/=np.linalg.norm(f)
       
      f=np.insert(f,0,1)
      
      X2=np.matmul(f,beta) 
        
      all_queries[t,:] = X2
      
    
    
    
  # feature normalization
  for i in range(int(all_queries.shape[0]/sz)):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(int(all_imgs.shape[0]/sz)):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  sims=np.zeros((1,int(all_imgs.shape[0]/sz)))

  for i in tqdm(range(int(all_queries.shape[0]/sz))):
    if (dot_eucld==0):
      sims = all_queries[i:(i+1), :].dot(all_imgs[:int(all_imgs.shape[0]/sz)].T)
    else:
      sims[0,:]=np.sum(abs(all_imgs[:int(all_imgs.shape[0]/sz),:]-all_queries[i, :]),axis=1)
      #for j in range(int(all_imgs.shape[0]/sz)):
      #  sims[0,j] =distance.euclidean( all_queries[i, :],all_imgs[j,:])


    if test_train==0:
      if (dot_eucld==0):
        sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      else:
        sims[0, test_queries[i]['source_img_id']] = 10e10  # remove query image
    if (dot_eucld==0):
      nn_result.append(np.argsort(-sims[0, :])[:105])
    else:
      nn_result.append(np.argsort(sims[0, :])[:105])

  all_imgs=[]
  all_queries=[]
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

  print(out)  
  
 
  return out

def train_network_on_saved(test_train,create_load,normal_normalize,filename,sz,dot_eucld):
  if test_train==0:
   with open(Path1+r"/"+'test_test_queries.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'test_all_queriesG.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'test_all_imgsG.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'rb') as fp:
    all_captions=pickle.load( fp)
   with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  else:
   with open(Path1+r"/"+'test_queries172k.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'all_queries172k.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'all_imgs172k.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   with open(Path1+r"/"+'all_captions172k.pkl', 'rb') as fp:
    all_captions=pickle.load( fp)
   with open(Path1+r"/"+'all_target_captions172k.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
    
    #################################
      
    
    
    
  # feature normalization
  for i in range(int(all_queries.shape[0]/sz)):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(int(all_imgs.shape[0]/sz)):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  sims=np.zeros((1,int(all_imgs.shape[0]/sz)))

  for i in tqdm(range(int(all_queries.shape[0]/sz))):
    if (dot_eucld==0):
      sims = all_queries[i:(i+1), :].dot(all_imgs[:int(all_imgs.shape[0]/sz)].T)
    else:
      sims[0,:]=np.sum(abs(all_imgs[:int(all_imgs.shape[0]/sz),:]-all_queries[i, :]),axis=1)
      #for j in range(int(all_imgs.shape[0]/sz)):
      #  sims[0,j] =distance.euclidean( all_queries[i, :],all_imgs[j,:])


    if test_train==0:
      if (dot_eucld==0):
        sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      else:
        sims[0, test_queries[i]['source_img_id']] = 10e10  # remove query image
    if (dot_eucld==0):
      nn_result.append(np.argsort(-sims[0, :])[:105])
    else:
      nn_result.append(np.argsort(sims[0, :])[:105])

  all_imgs=[]
  all_queries=[]
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

  print(out)  
  
 
  return out


class NLR2(nn.Module):
  def __init__(self,netin,netout,nethidden1):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.ReLU(),torch.nn.Linear(nethidden1, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

def build_and_train_net(hiddensize,max_iterations, min_error, all_queries,all_imgs,batch_size):
  all_queries=Variable(torch.Tensor(all_queries))
  all_imgs=variable(torch.tensor(all_imgs))
  model=NLR2(all_queries.shape[1],all_imgs.shape[1],hiddensize)
  #model=model.cuda()
  torch.manual_seed(3)
  loss_fn = torch.nn.MSELoss(reduction='sum')
  torch.manual_seed(3)
  criterion = nn.CosineSimilarity() 
  

  #loss.backward()

  #criterion=nn.MSELoss()
  optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
  epoch=max_iterations

  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    for l in range(int(all_queries.shape[0]/batch_size)):
      
      item_batch = all_queries[l*batch_size:(l+1)*batch_size-1,:]
      netoutbatch=model.myforward(item_batch)
      #loss=criterion(all_imgs[l*batch_size:(l+1)*batch_size-1,:],netoutbatch)
      loss = torch.mean(torch.abs(criterion(all_imgs[l*batch_size:(l+1)*batch_size-1,:],netoutbatch)))

      loss = 1 - loss

      losses.append(loss)
      #optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
      if (l%1000==0) :
        print('Epoch:',j,' get images batch=',l*batch_size,':',(l+1)*batch_size,'loss',loss,end='\r')
    if (total_loss<min_error):
      break
    print('iteration:',j, 'total loss',total_loss)
    totallosses.append(total_loss)

  print('Finished Training')
  torch.save(model.state_dict(), Path1+r'\NLP3.pth') 
  

def test_on_saved_NN_CMP(test_train,normal_beta_NN,create_load,filename,normal_normalize,sz,dot_eucld,hiddensize,model_fn):
  # test_queries:
  if test_train==0:
   with open(Path1+r"/"+'test_test_queries.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'test_all_queriesG.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'test_all_imgsG.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'rb') as fp:
    all_captions=pickle.load( fp)
   with open(Path1+r"/"+'test_all_target_captionsG.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  else:
   with open(Path1+r"/"+'test_queries172k.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'all_queries172k.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'all_imgs172k.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   with open(Path1+r"/"+'all_captions172k.pkl', 'rb') as fp:
    all_captions=pickle.load( fp)
   with open(Path1+r"/"+'all_target_captions172k.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  if (normal_beta_NN==2 ):
    ######### neural Network *********************************
    model=NLR2(all_queries.shape[1],all_imgs.shape[1],hiddensize)
    #torch.load(model.state_dict(), Path1+r'\NLP2.pth')
    #model.load_state_dict(torch.load(Path1+r'\'+NLP2.pth'))
    model.load_state_dict(torch.load(Path1+r'\''+model_fn))

    model.eval()
    all_queries=Variable(torch.Tensor(all_queries))

    for t in range(int(len(all_queries)/sz)):
      if (t%100==0):
        print('get testdata=',t,end='\r')
      f=all_queries[t,:]
    
      all_queries[t,:] = model.myforward(f)
    all_queries = torch.tensor(all_queries,requires_grad=False)
    #all_queries.detach().numpy()
    all_queries=np.array(all_queries)


    


  else:
    if (normal_beta_NN==1):
      if(create_load==0):
       #################################
        new_all_queries=np.zeros((all_queries.shape[0],all_queries.shape[1]+1))
        for i in range(all_queries.shape[0]):
          f=all_queries[i,:]
          if (normal_normalize==1):
            f/=np.linalg.norm(f)
          f=np.insert(f,0,1)
          new_all_queries[i,:]=f
          if (normal_normalize==1):
            for i in range(all_imgs.shape[0]):
              all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
      
        new_all_queriest=new_all_queries.transpose()
        X1=np.matmul(new_all_queriest,new_all_queries)  
        X2=np.linalg.inv(X1)
        X3=np.matmul(X2,new_all_queriest)  
        beta=np.matmul(X3,all_imgs) 
        new_all_queries=[]
        new_all_queriest=[]
        #################################
        with open(Path1+r"/"+filename, 'wb') as fp:
          pickle.dump( beta, fp)
      else:
        with open(Path1+r"/"+filename, 'rb') as fp:
          beta=pickle.load( fp)
      for t in range(int(len(all_queries)/sz)):
        if (t%100==0):
          print('get testdata=',t,end='\r')
        f=all_queries[t,:]
        if (normal_normalize==1):
         f/=np.linalg.norm(f)
       
        f=np.insert(f,0,1)
      
        X2=np.matmul(f,beta) 
        
        all_queries[t,:] = X2
      
    
    
    
  # feature normalization
  for i in range(int(all_queries.shape[0]/sz)):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(int(all_imgs.shape[0]/sz)):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  sims=np.zeros((1,int(all_imgs.shape[0]/sz)))

  for i in tqdm(range(int(all_queries.shape[0]/sz))):
    if (dot_eucld==0):
      sims = all_queries[i:(i+1), :].dot(all_imgs[:int(all_imgs.shape[0]/sz)].T)
    else:
      sims[0,:]=np.sum(abs(all_imgs[:int(all_imgs.shape[0]/sz),:]-all_queries[i, :]),axis=1)
      #for j in range(int(all_imgs.shape[0]/sz)):
      #  sims[0,j] =distance.euclidean( all_queries[i, :],all_imgs[j,:])


    if test_train==0:
      if (dot_eucld==0):
        sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
      else:
        sims[0, test_queries[i]['source_img_id']] = 10e10  # remove query image
    if (dot_eucld==0):
      nn_result.append(np.argsort(-sims[0, :])[:105])
    else:
      nn_result.append(np.argsort(sims[0, :])[:105])

  all_imgs=[]
  all_queries=[]
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

  print(out)  


#####################Loaded ################
def build_and_train_net_loaded(hiddensize,max_iterations, min_error,batch_size):
  all_imgs = datasets.Features172K().Get_all_images()
  all_captions = datasets.Features172K().Get_all_captions()
  all_queries = datasets.Features172K().Get_all_queries()
  all_target_captions = datasets.Features172K().Get_all_captions()
  
  model=NLR2(all_queries.shape[1],all_imgs.shape[1],hiddensize)
  #model=model.cuda()
  torch.manual_seed(3)
  loss_fn = torch.nn.MSELoss(reduction='sum')
  torch.manual_seed(3)
  criterion = nn.CosineSimilarity() 
  

  #loss.backward()

  #criterion=nn.MSELoss()
  optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
  epoch=max_iterations

  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    for l in range(int(all_queries.shape[0]/batch_size)):
      print('Epoch=',j,' Batch=',l,end='\r')      
      item_batch = all_queries[l*batch_size:(l+1)*batch_size-1,:]

      netoutbatch=model.myforward(torch.from_numpy(item_batch))
      #loss=criterion(all_imgs[l*batch_size:(l+1)*batch_size-1,:],netoutbatch)
      loss = torch.mean(torch.abs(criterion(torch.from_numpy(all_imgs[l*batch_size:(l+1)*batch_size-1,:]),netoutbatch)))

      loss = 1 - loss

      losses.append(loss)
      #optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss
      if (l%1000==0) :
        print('Epoch:',j,' get images batch=',l*batch_size,':',(l+1)*batch_size,'loss',loss,end='\r')
    if (total_loss<min_error):
      break
    print('iteration:',j, 'total loss',total_loss)
    totallosses.append(total_loss)

  print('Finished Training')
  torch.save(model.state_dict(), Path1+r'\NLP3.pth') 


def test_on_saved_NN_CMP_loaded(test_train):
  # test_queries:
  if test_train==0:
   

    all_imgs = datasets.Features33K().Get_all_images()
    all_captions = datasets.Features33K().Get_all_captions()
    all_queries = datasets.Features33K().Get_all_queries()
    all_target_captions = datasets.Features33K().Get_target_captions()
  else:
    all_imgs = datasets.Features172K().Get_all_images()[:10000]
    all_captions = datasets.Features172K().Get_all_captions()[:10000]
    all_queries = datasets.Features172K().Get_all_queries()[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions()[:10000]

  
  ######### neural Network *********************************
  model=NLR2(all_queries.shape[1],all_imgs.shape[1],700)
  #torch.load(model.state_dict(), Path1+r'\NLP2.pth')
  model.load_state_dict(torch.load(Path1+r'\NLP3.pth'))

  model.eval()
  all_queries=Variable(torch.Tensor(all_queries))

  # for t in range(int(len(all_queries))):
  #   if (t%100==0):
  #     print('get testdata=',t,end='\r')
  #   f=all_queries[t,:]
  
  #   all_queries[t,:] = model.myforward(f)

  all_queries = model.myforward(all_queries)
  all_queries = torch.tensor(all_queries,requires_grad=False)
  #all_queries.detach().numpy()
  all_queries=np.array(all_queries)

    
    
  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  nn_result = []
  for i in tqdm(range(all_queries.shape[0])):
    sims = all_queries[i:(i+1), :].dot(all_imgs.T)
    # if test_train==0:
    #   sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image
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

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  print(out)  




     
     


if __name__ == '__main__': 
    
  #with open(Path1+r"/"+'all_queries172k.pkl', 'rb') as fp:
  #  all_queries=pickle.load( fp)
  #with open(Path1+r"/"+'all_imgs172k.pkl', 'rb') as fp:
  #  all_imgs=pickle.load( fp)
  #build_and_train_net(700,1000, 50, all_queries,all_imgs,200)
  #def build_and_train_net(hiddensize,max_iterations, min_error, all_queries,all_imgs,batch_size):
  #test_on_saved_NN_CMP(test_train,normal_beta_NN,create_load,filename,normal_normalize,sz,dot_eucld,hiddensize):
  test_on_saved_NN_CMP(1,0,0,'nn',0,17.2,0,700,'')
  #test_on_saved_NN_CMP(0,0,0,'nn',0,1,0,700)
  test_on_saved_NN_CMP_loaded(0)
  test_on_saved_NN_CMP_loaded(1)
    

   
