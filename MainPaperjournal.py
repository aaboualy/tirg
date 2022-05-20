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

# class NLR3S(nn.Module):
#     def __init__(self,netin,netout,nethidden):
#       super().__init__()
#       self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden),torch.nn.Sigmoid(),torch.nn.Linear(nethidden, netout))
#     def myforward (self,inv):
#       outv=self.netmodel(inv)
#       return outv

# class NLR3T(nn.Module):
#     def __init__(self,netin,netout,nethidden):
#       super().__init__()
#       self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden),torch.nn.Tanh(),torch.nn.Linear(nethidden, netout))
#     def myforward (self,inv):
#       outv=self.netmodel(inv)
#       return outv

def Semantic50_Maa(run_type):
    device = torch.device("cpu")

    if run_type=='train':
        PhixQueryImg =datasets.Features50Org().phix_50[:50000]
        PhitQueryCaption =datasets.Feature172KOrg().PhitQueryCaption[:50000]
        PhitQueryMod =datasets.Feature172KOrg().PhitQueryMod[:50000]
        PhixTargetImg =datasets.Features50Org().target_phix_50[:50000]
        PhitTargetCaption =datasets.Feature172KOrg().PhitTargetCaption[:50000]
        all_captions_text =datasets.Feature172KOrg().all_captions_text[:50000]
        all_target_captions_text =datasets.Feature172KOrg().all_target_captions_text[:50000]
        all_Query_captions_text =datasets.Feature172KOrg().all_Query_captions_text[:50000]
        all_ids =datasets.Feature172KOrg().all_ids[:50000]
        #SearchedFeatures=PhixTargetImg


    elif run_type=='test':
        PhixQueryImg=datasets.Features50Org().phix_50_test
        PhitQueryCaption=datasets.Features33KOrg().PhitQueryCaption
        PhitQueryMod=datasets.Features33KOrg().PhitQueryMod
        PhixTargetImg=datasets.Features50Org().target_phix_50_test
        PhitTargetCaption=datasets.Features33KOrg().PhitTargetCaption
        PhixAllImages=datasets.Features33KOrg().PhixAllImages
        PhitAllImagesCaptions=datasets.Features33KOrg().PhitAllImagesCaptions
        all_captions_text=datasets.Features33KOrg().all_captions_text
        all_target_captions_text=datasets.Features33KOrg().all_target_captions_text
        all_Query_captions_text=datasets.Features33KOrg().all_queries_captions_text
        all_queries_Mod_text=datasets.Features33KOrg().all_queries_Mod_text
        all_ids=datasets.Features33KOrg().all_ids
        #SearchedFeatures=PhixAllImages


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

def euclideandistance(signature,signatureimg):
    from scipy.spatial import distance
    return distance.euclidean(signature, signatureimg)

def SaveFeaturesToFiles():
    #datasets.FeaturesToFiles172().SaveAllFeatures()
    datasets.FeaturesToFiles33().SaveAllFeatures()

def ValidateFeaturesToFiles():
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

    for i in range(172048):#172048
      print('Validate:',i,end='\r')
      item = train[i]
      datasets.FeaturesToFiles172().ValidateFile(item['source_img_id'],trig)
      datasets.FeaturesToFiles172().ValidateFile(item['target_img_id'],trig)

    test_queries = test.get_test_queries()
    for item in tqdm(test_queries):
        datasets.FeaturesToFiles33().ValidateFile(item['source_img_id'],trig)
        datasets.FeaturesToFiles33().ValidateFile(item['target_id'],trig)

def ValidateFile172():
    Path=Path1+r'/FeaturesToFiles172'
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

    with open (Path+r'/Features172QueryStructureallF.txt', 'rb') as fp:
      allinone = pickle.load(fp)


    for i in range(len(allinone)):
        item = allinone[i]
        idx= item['QueryID']

        with open (Path+r'/FeaturesToFiles172.txt', 'rb') as fp:
            Idximgs = pickle.load(fp)

        print('Img in Index of Dataset:',train.imgs[idx])
        print('Img in Index from File:',Idximgs[idx])

        imgF = [trig.extract_img_feature(torch.stack([train.get_img(idx)]).float()).data.cpu().numpy()]
        text_modelF = [trig.extract_text_feature([train.imgs[idx]['captions'][0]]).data.cpu().numpy()]

        print('Caption=',train.imgs[idx]['captions'][0])
        print('Caption=',item['QueryCaption']) 

    
        Resnet152 = models.resnet152(pretrained=True)
        Resnet152.fc = nn.Identity()
        Resnet152.eval()

        Resnet50 = models.resnet50(pretrained=True)
        Resnet50.fc = nn.Identity()
        Resnet50.eval()

        Resnet18 = models.resnet18(pretrained=True)
        Resnet18.fc = nn.Identity()
        Resnet18.eval()


        img=train.get_img(idx)
        img=torch.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))


        out=Resnet152(img)
        out = Variable(out, requires_grad=False)
        out=np.array(out)
        Feature152 =[out[0,:]]

        out=Resnet50(img)
        out = Variable(out, requires_grad=False)
        out=np.array(out)
        Feature50 =[out[0,:]]

        out=Resnet18(img)
        out = Variable(out, requires_grad=False)
        out=np.array(out)
        Feature18 =[out[0,:]]

        print ('Distance Between img Tirg:', euclideandistance(imgF,item['QuerytrigF']))
        print ('Distance Between text Tirg:', euclideandistance(text_modelF,item['QueryCaptionF']))
        print ('Distance Between img 18:', euclideandistance(Feature18,item['Query18F']))
        print ('Distance Between img 50:', euclideandistance(Feature50,item['Query50F']))
        print ('Distance Between img 152:', euclideandistance(Feature152,item['Query152F']))
       
def ValidateFile33():
    Path=Path1+r'/FeaturesToFiles33'

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

    with open (Path+r'/Features33QueryStructureallF.txt', 'rb') as fp:
      allinone = pickle.load(fp)


    for i in range(len(allinone)):
        item = allinone[i]
        idx= item['QueryID']

        with open (Path+r'/FeaturesToFiles33.txt', 'rb') as fp:
            Idximgs = pickle.load(fp)

        print('Img in Index of Dataset:',test.imgs[idx])
        print('Img in Index from File:',Idximgs[idx])

        imgF = [trig.extract_img_feature(torch.stack([test.get_img(idx)]).float()).data.cpu().numpy()]
        text_modelF = [trig.extract_text_feature([test.imgs[idx]['captions'][0]]).data.cpu().numpy()]

        print('Caption=',test.imgs[idx]['captions'][0])
        print('Caption=',item['QueryCaption']) 

    
        Resnet152 = models.resnet152(pretrained=True)
        Resnet152.fc = nn.Identity()
        Resnet152.eval()

        Resnet50 = models.resnet50(pretrained=True)
        Resnet50.fc = nn.Identity()
        Resnet50.eval()

        Resnet18 = models.resnet18(pretrained=True)
        Resnet18.fc = nn.Identity()
        Resnet18.eval()


        img=test.get_img(idx)
        img=torch.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))


        out=Resnet152(img)
        out = Variable(out, requires_grad=False)
        out=np.array(out)
        Feature152 =[out[0,:]]

        out=Resnet50(img)
        out = Variable(out, requires_grad=False)
        out=np.array(out)
        Feature50 =[out[0,:]]

        out=Resnet18(img)
        out = Variable(out, requires_grad=False)
        out=np.array(out)
        Feature18 =[out[0,:]]

        print ('Distance Between img Tirg:', euclideandistance(imgF,item['QuerytrigF']))
        print ('Distance Between text Tirg:', euclideandistance(text_modelF,item['QueryCaptionF']))
        print ('Distance Between img 18:', euclideandistance(Feature18,item['Query18F']))
        print ('Distance Between img 50:', euclideandistance(Feature50,item['Query50F']))
        print ('Distance Between img 152:', euclideandistance(Feature152,item['Query152F']))
 
def Semantic152_Maa(run_type):
    device = torch.device("cpu")

    if run_type=='train':
        with open (Path1+r'/FeaturesToFiles172/Features172QueryStructureallF.txt', 'rb') as fp:
            AllData = pickle.load(fp) 

        PhixQueryImg =[d['Query152F'] for d in AllData[:10000]]
        PhitQueryCaption =[d['QueryCaptionF'] for d in AllData[:10000]]
        PhitQueryMod =[d['ModF'][0] for d in AllData[:10000]]
        PhixTargetImg =[d['Target152F'] for d in AllData[:10000]]
        PhitTargetCaption =[d['TargetCaptionF'] for d in AllData[:10000]]
        all_captions_text =[d['TargetCaption'] for d in AllData[:10000]]
        all_target_captions_text =[d['TargetCaption'] for d in AllData[:10000]]
        all_Query_captions_text =[d['QueryCaption'] for d in AllData[:10000]]
        
        


    elif run_type=='test':
        with open (Path1+r'/FeaturesToFiles33/Features33QueryStructureallF.txt', 'rb') as fp:
            AllData = pickle.load(fp) 

        PhixQueryImg =[d['Query152F'] for d in AllData]
        PhitQueryCaption =[d['QueryCaptionF'] for d in AllData]
        PhitQueryMod =[d['ModF'][0] for d in AllData]
        PhixTargetImg =[d['Target152F'] for d in AllData]
        PhitTargetCaption =[d['TargetCaptionF'] for d in AllData]
        all_captions_text =[d['TargetCaption'] for d in AllData]
        all_target_captions_text =[d['TargetCaption'] for d in AllData]
        all_Query_captions_text =[d['QueryCaption'] for d in AllData]


    PhitQueryCaption=torch.tensor(PhitQueryCaption).to(device)
    PhixQueryImg=torch.tensor(PhixQueryImg).to(device)
    PhitQueryMod=torch.tensor(PhitQueryMod).to(device)
    PhitTargetCaption=torch.tensor(PhitTargetCaption).to(device)
    PhixTargetImg=torch.tensor(PhixTargetImg).to(device)

    phix=PhixQueryImg
    phit=PhitQueryMod



    hidden=1000
    NetA=NLR3T(phix.shape[1],phit.shape[1],hidden)
    NetA.load_state_dict(torch.load( Path1+r'/UltraNetA152.pth', map_location=torch.device('cpu') ))
    hidden=2500
    NetB=NLR3T(phit.shape[1],phix.shape[1],hidden)
    NetB.load_state_dict(torch.load( Path1+r'/UltraNetB152_2500.pth', map_location=torch.device('cpu') ))
    hidden=1800
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],hidden)
    NetC.load_state_dict(torch.load( Path1+r'/ulteraNetC.pth', map_location=torch.device('cpu') ))



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







########################################################################################################################################




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
        
        #AllData=AllData[:10000]
    
    elif run_test==1:
        with open (Path1+r'/FeaturesToFiles33/Features33QueryStructureallF.txt', 'rb') as fp:
            AllData = pickle.load(fp) 

    
    phix=[d['Query18F'] for d in AllData]
    phit=[d['ModF'] for d in AllData]
    phix=torch.tensor(phix)
    phit=torch.tensor(phit)
    phit=torch.squeeze(phit)
    target=[d['Target18F'] for d in AllData]
    target=torch.tensor(target)

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
    del AllData
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
        sims = net_target[i, :].dot(phix[:net_target.shape[0],:].T)
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
        sims = net_target[i, :].dot(phix[:net_target.shape[0],:].T)
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
    print(flags)
    print (out)

def Semantic152_5(run_test):
    if run_test==0:
        with open(Path1+r'/FeaturesToFiles172/Features172QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)
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

        if run_test==0 :
            NetA_target=[d['QueryCaptionF'] for d in AllData]
            NetC_target=[d['TargetCaptionF'] for d in AllData]
            NetA_target=torch.tensor(NetA_target)
            NetC_target=torch.tensor(NetC_target)
        del AllData
    
    hidden=1000
    NetA=NLR3T(phix.shape[1],phit.shape[1],hidden)
    # Final_NetANN152 UltraNetA152 ulteraNetA_2500_UTS UltraNetA152tune ulteraNetA_Ujn UltraNetA152tune
    NetA.load_state_dict(torch.load( Path1+r'/UltraNetA152tunesntc.pth', map_location=torch.device('cpu') ))
    hidden=2500
    NetB=NLR3T(phit.shape[1],phix.shape[1],hidden)
    #NetB_2500_UUTS2   UltraNetB152_2500 UltraNetB152_CO25042final UltraNetB152_CO25042 UltraNetB152_CO25042final2
    NetB.load_state_dict(torch.load( Path1+r'/UltraNetB152_CO25042final2.pth', map_location=torch.device('cpu') ))
    hidden=1800
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],hidden)
    # Final_ulteraNetC ulteraNetC_best
    NetC.load_state_dict(torch.load( Path1+r'/Final_ulteraNetC.pth', map_location=torch.device('cpu') ))

    if run_test==0:
        ACloss_fn=torch.nn.MSELoss()
        Bloss_fn=torch.nn.CosineSimilarity()
        NetAout=NetA.myforward(phix)

        Aloss=ACloss_fn(NetAout,NetA_target)
        print('loss A  ',Aloss)
        NetCinp=torch.cat((phit,NetAout),1)
        NetCout=NetC.myforward(NetCinp)
    

        if run_test==0:
            del NetCinp, NetAout, NetA_target

        Closs=ACloss_fn(NetCout,NetC_target)
        print('loss c  ',Closs)
        net_target=NetB.myforward(NetCout)

        if run_test==0:
            del NetCout, NetC_target
        Bloss=torch.mean(Bloss_fn(net_target[:33000],target[:33000]))
        print('loss   B',Bloss)
    else:
        Bloss_fn=torch.nn.CosineSimilarity()
        NetAout=NetA.myforward(phix)
        NetCinp=torch.cat((phit,NetAout),1)
        NetCout=NetC.myforward(NetCinp)
       
        del NetCinp, NetAout
        net_target=NetB.myforward(NetCout)
        del NetCout
        Bloss=torch.mean(Bloss_fn(net_target,target))
        print('loss   B',Bloss)

    if (run_test==0):
        with open(Path1+r'/FeaturesToFiles172/Features172QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)
    else:
        with open(Path1+r'/FeaturesToFiles33/Features33QueryStructureallF.txt', 'rb') as fp:
            AllData=pickle.load( fp)
  

    #allquerycaptions=[d['QueryCaption'] for d in AllData]
    alltargetcaptions=[d['TargetCaption'] for d in AllData]
    del AllData
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

    print(phix.shape)

    nn_result = []
    #phixN=torch.tensor(phixN)
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    for i in range(phix.shape[0]):
        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])

    for i in range (20000):  #(3900): #
        sims = net_target[i, :].dot(phix[:net_target.shape[0],:].T)
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
    #Semantic18_5(1)
    

    
   

    

