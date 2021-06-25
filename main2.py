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
#import datasets
from tqdm import tqdm as tqdm
import PIL
import argparse
import datasets
import img_text_composition_models




#Path1=r"D:\personal\master\MyCode\files"
Path1=r"C:\MMaster\Files"


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
  
  for name, dataset in [ ('test', test),('train', train)]: #('train', trainset),
  #for name, dataset in [ ('test', test)]: #('train', trainset),
          
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
    print('       ', sourceFile)
  if (dot_eucld==0):
    print(' Distance: Cos Angle between vectors ', file = sourceFile)
  else:
    print(' Distance: Eucledian  ', file = sourceFile)
  print(' Dataset size Divider ', set_size_divider, file = sourceFile)
  print(' Experiment Outcome: - ','\n',out,'\n', file = sourceFile)

def results():
  stime=time.strftime("%Y%m%d-%H%M%S")
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'w')
  test_train=1
  normal_beta=0
  set_size_divider=1
  normal_normalize=0
  create_load=0
  filename='na'
  dot_eucld=0
  # 1
  print(' 1 ', file=sourceFile)
  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=17.2
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')


  # 2
  print(' 2', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  normal_beta=1
  create_load=0
  filename='REGTR10ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 3
  print(' 3', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 4
  print(' 4', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTS33ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 5
  # print(' 5', sourceFile)

  # out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  # print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  test_train=1
  set_size_divider=1
  normal_beta=0
  create_load=0
  filename='na'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 6
  # print(' 6', sourceFile)

  # out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  # print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)

  test_train=1
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTR172ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 7
  # print(' 7', sourceFile)

  # out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  # print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
 
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR172ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

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
  
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  
  # 3NN
  print(' 3NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10NND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 4 NN
  print(' 4NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTR172NND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 7 NN
  print(' 7NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
 
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR172NND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 8 NN
  print(' 8NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  test_train=1
  set_size_divider=1
  normal_beta=0
  create_load=0
  filename='na'
  dot_eucld=1
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 1 E
  print(' 1 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=17.2
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 2 E
  print(' 2 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  normal_beta=1
  create_load=0
  filename='REGTR10NE.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 3 E
  print(' 3 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10NE.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')
  
  # 4 E
  print(' 4 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTS33NE.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 5 E
  print(' 5 E', file=sourceFile)


def SaveFilesFeatures():
  
  
  test_train=1
  set_size_divider=1
  normal_beta=0
  create_load=0
  filename='na'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 6 E
  print(' 6 E', file=sourceFile)

  

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
  

def results2():
  stime='20210615-072642'
 # sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'w')
  test_train=0
  normal_beta=0
  set_size_divider=1
  normal_normalize=0
  create_load=0
  filename='na'
  dot_eucld=0
  # 1
  #print(' 1', file=sourceFile)
  #out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=17.2
  #sourceFile.close()
  #sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')


  # 2
  #print(' 2', file=sourceFile)

  #out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  normal_beta=1
  create_load=0
  filename='REGTR10ND.BTA'
  #sourceFile.close()
  #sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 3
  #print(' 3', file=sourceFile)

  #out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10ND.BTA'
  #sourceFile.close()
  #sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 4
  #print(' 4', file=sourceFile)

  #out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTS33ND.BTA'
  #sourceFile.close()
  #sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 5
  #print(' 5', file=sourceFile)

  #out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  test_train=1
  set_size_divider=10
  normal_beta=0
  create_load=0
  filename='na'
  #sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 6
  print(' 6', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)

  test_train=1
  set_size_divider=10
  normal_beta=1
  create_load=0
  filename='REGTR172ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 7
  print(' 7', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
 
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR172ND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 8
  print(' 8', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  ###################NNORMALIZED BETA##############################################################
  test_train=1
  normal_beta=1
  set_size_divider=17.2
  normal_normalize=0
  create_load=0
  filename='REGTR10NND.BTA'
  dot_eucld=0
  test_train=1
  
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  
  # 3NN
  print(' 3NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10NND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 4 NN
  print(' 4NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=10
  normal_beta=1
  create_load=0
  filename='REGTR172NND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 7 NN
  print(' 7NN', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
 
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR172NND.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

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
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 1 E
  print(' 1 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  set_size_divider=17.2
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 2 E
  print(' 2 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=1
  normal_beta=1
  create_load=0
  filename='REGTR10NE.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 3 E
  print(' 3 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=1
  filename='REGTR10NE.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')
  
  # 4 E
  print(' 4 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  test_train=0
  set_size_divider=1
  normal_beta=1
  create_load=0
  filename='REGTS33NE.BTA'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 5 E
  print(' 5 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  
  test_train=1
  set_size_divider=10
  normal_beta=0
  create_load=0
  filename='na'
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  # 6 E
  print(' 6 E', file=sourceFile)

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)

  sourceFile.close()

def results_temp():
  stime=time.strftime("%Y%m%d-%H%M%S")
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'w')
  test_train=1
  normal_beta=0
  set_size_divider=10
  normal_normalize=0
  create_load=0
  filename='beta1806.pkl'
  dot_eucld=0
  # 1
  print(' 1 ', file=sourceFile)
  print ('1')
  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')
  test_train=1
  normal_beta=1
  set_size_divider=10
  normal_normalize=1
  create_load=0
  filename='beta1806.pkl'
  dot_eucld=0
  

  # 2
  print(' 2', file=sourceFile)
  print('2')

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  test_train=1
  normal_beta=0
  set_size_divider=10
  normal_normalize=0
  create_load=0
  filename='beta1806euc.pkl'
  dot_eucld=1
  # 1
  print(' 3 ', file=sourceFile)
  print ('3 ')
  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')
  test_train=1
  normal_beta=1
  set_size_divider=10
  normal_normalize=1
  create_load=0
  filename='beta1806euc.pkl'
  dot_eucld=1
  

  # 2
  print(' 4', file=sourceFile)
  print('4')

  out =test_retrieval.test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

 

  datasets.Features172K().SavetoFiles(Path1+r'/dataset172', trig, trainset,opt)
  print('172 Finished')
  datasets.Features33K().SavetoFiles(Path1+r'/dataset33', trig, testset,opt)
  print('33 Finished')

def getvaluesfilesaved():

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
    
    asbook1 = test_retrieval.testLoaded(opt, trig, dataset)
    print(name,' Loaded As PaPer: ',asbook1)

    asbook = test_retrieval.test(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)
     
class NLR2(nn.Module):
  def __init__(self,netin,netout,nethidden1):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.ReLU(),torch.nn.Linear(nethidden1, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

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

def NLP2Values():

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
    
    asbook1 = test_retrieval.testLoaded_NLP(opt, trig, dataset)
    print(name,' NLP2 Loaded : ',asbook1)

    asbook = test_retrieval.testLoaded(opt, trig, dataset)
    print(name,' Loaded As PaPer: ',asbook)
    

if __name__ == '__main__': 
  getvaluesfilesaved()
  #NLP2Values()
  #getvaluesfilesaved()
    
  #getbetatrain()
  # GetValuestrain()
  #savevaluestofile()
  #Savevaluestest()
  #Savevaluestest()
  #Save_GetValues()
  #results_temp()
  #asbook = test_retrieval.test_on_saved(1,0)
  #print('train',' As PaPer: ',asbook)
  #asbook = test_retrieval.test_on_saved(0,0)
  #print('test',' As PaPer: ',asbook)
  #results2()
  
  #asbook = test_retrieval.test_on_saved(0,1)
  #print('test
  # ',' As PaPer: ',asbook)
  #test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #getbeta()
  #Save_GetValues()

    

   
