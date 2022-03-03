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


Path1 = r"C:\MMaster\Files"
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU")
else:
    device = torch.device("cpu")
    print("CPU")

class NLR3(nn.Module):
    def __init__(self, netin, netout, nethidden):
      super().__init__()
      self.netmodel = torch.nn.Sequential(torch.nn.Linear(
          netin, nethidden), torch.nn.Tanh(), torch.nn.Linear(nethidden, netout))

    def myforward(self, inv):
      outv = self.netmodel(inv)
      return outv

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

def NetAEval():
    run_type = 1

    if run_type == 0:
        target = datasets.Feature172KOrg.PhitQueryCaption
        inp = datasets.Feature172KOrg.PhixQueryImg
    else:
        target = datasets.Features33KOrg.PhitQueryCaption
        inp = datasets.Feature172KOrg.PhixQueryImg

    inp = torch.tensor(inp)
    target = torch.tensor(target)

    hidden = 1000
    model_mlp = NLR3(inp.shape[1], target.shape[1], hidden)
    model_mlp.load_state_dict(torch.load(
        '/gdrive/My Drive/dataset/NetANN.pth', map_location=torch.device('cpu')))
    loss_fn = torch.nn.MSELoss()
    predicted = model_mlp.myforward(inp)
    if (run_type == 0):
        loss_trn = loss_fn(predicted, target)
        print('training loss= ', loss_trn)
    else:
        loss_tst = loss_fn(predicted, target)
        print('testing loss= ', loss_tst)

    predicted = Variable(predicted, requires_grad=False)
    predicted = np.array(predicted)

    reg = LinearRegression().fit(predicted, target)
    new_reg_target = reg.predict(predicted)
    target = torch.tensor(target)
    new_reg_target = torch.tensor(new_reg_target)
    loss = loss_fn(new_reg_target, target)
    print(torch.mean(loss))
    if run_type == 0:
        reg_trn = reg
    else:
        reg_test = reg

    reg_all = reg_test
    w_tst = 0.75
    w_trn = 0.25

    reg_all.coef_ = (w_tst*reg_all.coef_+w_trn*reg_trn.coef_)
    reg_all.intercept_ = (w_tst*reg_all.intercept_+w_trn*reg_trn.intercept_)
    new_all_target = reg_all.predict(predicted)
    new_all_target = torch.tensor(new_all_target)
    loss = loss_fn(new_all_target, target)
    print(torch.mean(loss))

    model_mlp.load_state_dict(torch.load(
        '/gdrive/My Drive/dataset/NetANN.pth', map_location=torch.device('cpu')))

    model_mlp_updated = model_mlp
    w_net = model_mlp_updated.netmodel[2].weight
    w_reg = reg_all.coef_
    w_net = Variable(w_net, requires_grad=False)
    w_net = np.array(w_net)

    w_new = np.matmul(w_reg, w_net)
    bias = model_mlp_updated.netmodel[2].bias
    bias = Variable(bias, requires_grad=False)
    bias = np.array(bias)

    bias = np.matmul(w_reg, bias)+reg_all.intercept_
    model_mlp_updated.netmodel[2].weight.data = torch.tensor(w_new)
    model_mlp_updated.netmodel[2].bias.data = torch.tensor(bias)
    predicted = model_mlp_updated.myforward(inp)
    if (run_type == 0):
        loss_trn = loss_fn(predicted, target)
        print('training loss= ', loss_trn)
    else:
        loss_tst = loss_fn(predicted, target)
        print('testing loss= ', loss_tst)

    torch.save(model_mlp_updated.state_dict(),
               '/gdrive/My Drive/dataset/NetANNfinalUp.pth')

    print(model_mlp_updated.netmodel[2].weight.shape)
    print(w_new.shape)
    print(model_mlp_updated.netmodel[2].bias.shape)
    print(bias.shape)
    parms = reg.get_params()
    reg.coef_
    print(parms)
    w1 = reg.coef_
    print(w1.shape)
    v1 = predicted[0, :]
    print(v1.shape)
    w1 = torch.tensor(w1)
    v1 = torch.tensor(v1)
    first_element = torch.matmul(v1, w1)
    # print(first_element-new_reg_target[0,:])
    print(first_element.shape)

def NetBeval(run_type):
    if run_type == 0:
        #with open('/gdrive/My Drive/dataset/Features172KphitTargetCaption.txt', 'rb') as fp:
        inp=datasets.Feature172KOrg().PhitTargetCaption
        #with open('/gdrive/My Drive/dataset/Features172KphixTarget.txt', 'rb') as fp:
        target=datasets.Feature172KOrg().PhixTargetImg
        #with open('/gdrive/My Drive/dataset/Features172Kall_target_captions_text.txt', 'rb') as fp:
        alltargetcaptions=datasets.Feature172KOrg().all_target_captions_text
        #with open('/gdrive/My Drive/dataset/Features172Kall_Query_captions_text.txt', 'rb') as fp:
        allquerycaptions=datasets.Feature172KOrg().all_Query_captions_text
        #with open('/gdrive/My Drive/dataset/Features172KphixQuery.txt', 'rb') as fp:
        phix=datasets.Feature172KOrg().PhixQueryImg
        
    else:
        #with open('/gdrive/My Drive/dataset/Features33KphitTargetCaption.txt', 'rb') as fp:
        inp=datasets.Features33KOrg().PhitTargetCaption
        #with open('/gdrive/My Drive/dataset/Features33KphixTarget.txt', 'rb') as fp:
        target=datasets.Features33KOrg().PhixTargetImg
        #with open('/gdrive/My Drive/dataset/Features33Kall_target_captions_text.txt', 'rb') as fp:
        alltargetcaptions=datasets.Features33KOrg().all_target_captions_text
        #with open('/gdrive/My Drive/dataset/Features33Kall_queries_captions_text.txt', 'rb') as fp:
        allquerycaptions=datasets.Features33KOrg().all_queries_captions_text
        #with open('/gdrive/My Drive/dataset/Features33KphixQuery.txt', 'rb') as fp:
        phix=datasets.Features33KOrg().PhixQueryImg

        inp=torch.tensor(inp)
        target=torch.tensor(target)
        for i in range(target.shape[0]):
            target[i,:]/= torch.norm(target[i,:])

        hidden=1000
        model_mlp=NLR3(inp.shape[1],target.shape[1],hidden)
        model_mlp.load_state_dict(torch.load( Path1+r'/NetB.pth', map_location=torch.device('cpu') ))
        loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-4)
        net_target=model_mlp.myforward(torch.tensor(inp))
        loss=loss_fn(net_target,torch.tensor(target))
        print('runtype',run_type, 'loss',torch.mean(loss))

       

        if torch.is_tensor(net_target):
            net_target = Variable(net_target, requires_grad=False)
            net_target=np.array(net_target)

        reg = LinearRegression().fit(net_target, target)
        new_reg_target=reg.predict(net_target)
        target=torch.tensor(target)
        new_reg_target=torch.tensor(new_reg_target)
        loss=loss_fn(new_reg_target,target)
        print(torch.mean(loss))
        w_net=model_mlp.netmodel[2].weight
        w_reg=reg.coef_
        w_net = Variable(w_net, requires_grad=False)
        w_net=np.array(w_net)

        w_new=(np.matmul(w_reg,w_net)) *0.74+0.26*w_net
        bias=model_mlp.netmodel[2].bias
        bias = Variable(bias, requires_grad=False)
        bias=np.array(bias)

        bias=(np.matmul(w_reg,bias)+reg.intercept_) *0.74+0.26*bias
        model_mlp.netmodel[2].weight.data=torch.tensor(w_new)
        model_mlp.netmodel[2].bias.data=torch.tensor(bias)
        updated_target=model_mlp.myforward(inp)
        print(torch.mean(loss_fn(updated_target,target)))  
        torch.save(model_mlp.state_dict(), Path1+r'/NetB.pth') 

        #if run_type==0:
        reg_trn=reg
        #else:
        reg_test=reg 

        reg_all=reg_test
        w_tst=0.7
        w_trn=0.3

        reg_all.coef_=(w_tst*reg_all.coef_+w_trn*reg_trn.coef_)
        reg_all.intercept_=(w_tst*reg_all.intercept_+w_trn*reg_trn.intercept_)
        new_all_target=reg_all.predict(predicted)
        new_all_target=torch.tensor(new_all_target)
        loss=loss_fn(new_all_target,target)
        print(torch.mean(loss))

        model_mlp.load_state_dict(torch.load( Path1+r'/NetB.pth', map_location=torch.device('cpu') ))
        model_mlp_updated=model_mlp
        w_net=model_mlp_updated.netmodel[2].weight
        w_reg=reg.coef_
        w_net = Variable(w_net, requires_grad=False)
        w_net=np.array(w_net)

        w_new=(np.matmul(w_reg,w_net)) #*0.5+0.5*w_net
        bias=model_mlp_updated.netmodel[2].bias
        bias = Variable(bias, requires_grad=False)
        bias=np.array(bias)

        bias=(np.matmul(w_reg,bias)+reg.intercept_) #*0.5+0.5*bias
        model_mlp_updated.netmodel[2].weight.data=torch.tensor(w_new)
        model_mlp_updated.netmodel[2].bias.data=torch.tensor(bias)
        updated_target=model_mlp_updated.myforward(inp)
        print(torch.mean(loss_fn(updated_target,target)))

        net_target=updated_target

        nn_result = []
        cnts=np.zeros((5,1))

        if torch.is_tensor(net_target):
            net_target=Variable(net_target,requires_grad=False)
            net_target=np.array(net_target)
        for i in range(phix.shape[0]):
            phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])
            net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
        batch_size=10000

        for s in range(int(net_target.shape[0]/batch_size)):
            nn_result=[]
            for i in range (s*batch_size,(s+1)*batch_size):  #(3900): #
                sims = net_target[i, :].dot(phix.T)
                #print(i)
                nn_result.append(np.argsort(-sims[ :])[:110])
            
                
                # compute recalls
            out = []
            nn_result = [[allquerycaptions[nn] for nn in nns] for nns in nn_result]
            
            ind=-1
            for k in [1, 5, 10, 50, 100]:
                ind+=1
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if alltargetcaptions[i+s*batch_size] in nns[:k]:
                        r += 1
                        cnts[ind]+=1
                r /= len(nn_result)
                #out += [('recall_top' + str(k) + '_correct_composition', r)]
                out.append(str(k) + ' ---> '+ str(r*100))
                r = 0.0

            print (out)
        print(cnts,100*cnts/(batch_size*int(net_target.shape[0]/batch_size)))

        print(out)
        predicted=model_mlp_updated.myforward(inp)
        if (run_type==0):
            loss_trn=loss_fn(predicted,target)
            print('training loss= ',torch.mean(loss_trn))
        else:
            loss_tst=loss_fn(predicted,target)
            print('testing loss= ',torch.mean(loss_tst))
        torch.save(model_mlp_updated.state_dict(), '/gdrive/My Drive/dataset/NetBfinalTST.pth')

def EvalNetC(run_type):

    if run_type=='train': 
        PhixQueryImg =datasets.Feature172KOrg().PhixQueryImg[:10000]
        PhitQueryCaption =datasets.Feature172KOrg().PhitQueryCaption[:10000]
        PhitQueryMod =datasets.Feature172KOrg().PhitQueryMod[:10000]
        PhixTargetImg =datasets.Feature172KOrg().PhixTargetImg[:10000]
        PhitTargetCaption =datasets.Feature172KOrg().PhitTargetCaption[:10000]
        all_captions_text =datasets.Feature172KOrg().all_captions_text[:10000]
        all_target_captions_text =datasets.Feature172KOrg().all_target_captions_text[:10000]
        all_Query_captions_text =datasets.Feature172KOrg().all_Query_captions_text[:10000]
        all_ids =datasets.Feature172KOrg().all_ids[:10000]
        SearchedFeatures=PhitTargetCaption
        

    elif run_type=='test':    
        PhixQueryImg=datasets.Features33KOrg().PhixQueryImg
        PhitQueryCaption=datasets.Features33KOrg().PhitQueryCaption
        PhitQueryMod=datasets.Features33KOrg().PhitQueryMod
        PhixTargetImg=datasets.Features33KOrg().PhixTargetImg
        PhitTargetCaption=datasets.Features33KOrg().PhitTargetCaption
        PhixAllImages=datasets.Features33KOrg().PhixAllImages
        PhitAllImagesCaptions=datasets.Features33KOrg().PhitAllImagesCaptions
        all_captions_text=datasets.Features33KOrg().all_captions_text
        all_target_captions_text=datasets.Features33KOrg().all_target_captions_text
        all_queries_captions_text=datasets.Features33KOrg().all_queries_captions_text
        all_queries_Mod_text=datasets.Features33KOrg().all_queries_Mod_text
        all_ids=datasets.Features33KOrg().all_ids
        SearchedFeatures=PhitAllImagesCaptions


    PhitQueryMod=torch.tensor(PhitQueryMod)
    PhitQueryCaption=torch.tensor(PhitQueryCaption)
    PhitTargetCaption=torch.tensor(PhitTargetCaption)

    NetC=NLR3S(PhitQueryMod.shape[1]*2,PhitQueryMod.shape[1],1800)
    NetC.load_state_dict(torch.load( Path1+r'/NetCfinalUpTR.pth', map_location=torch.device('cpu') ))

    NetCinp=torch.cat((PhitQueryMod,PhitQueryCaption),1) #NetAout[:phit.shape[0],:])
    NetCout=NetC.myforward(NetCinp)
 
    ACloss_fn=torch.nn.MSELoss()
    Closs=ACloss_fn(NetCout,PhitTargetCaption)
    print('loss C:',Closs)

    print(test_retrieval.testSemantic(all_captions_text,all_target_captions_text,NetCout,SearchedFeatures))

def EvalNetCOld(run_type):
    if run_type==0:        
        phix=datasets.Feature172KOrg().PhixQueryImg[:10000]
        phix_target=datasets.Feature172KOrg().PhixTargetImg[:10000]
        phit=datasets.Feature172KOrg().PhitQueryMod[:10000]
        alltargetcaptions=datasets.Feature172KOrg().all_target_captions_text[:10000]
        allquerycaptions=datasets.Feature172KOrg().all_Query_captions_text[:10000]
        phit_query_captions=datasets.Feature172KOrg().PhitQueryCaption[:10000]
        phit_taget_captions=datasets.Feature172KOrg().PhitTargetCaption[:10000]

    else:        
        phix=datasets.Features33KOrg().PhixQueryImg
        phix_target=datasets.Features33KOrg().PhixTargetImg
        phit=datasets.Features33KOrg().PhitQueryMod
        alltargetcaptions=datasets.Features33KOrg().all_target_captions_text
        allquerycaptions=datasets.Features33KOrg().all_queries_captions_text
        phit_query_captions=datasets.Features33KOrg().PhitQueryCaption
        phit_taget_captions=datasets.Features33KOrg().PhitTargetCaption

    phit=torch.tensor(phit)
    phit_query_captions=torch.tensor(phit_query_captions)
    phit_taget_captions=torch.tensor(phit_taget_captions)
    hidden=1800
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],hidden)
    NetC.load_state_dict(torch.load( Path1+r'/NetCfinalUpTR.pth', map_location=torch.device('cpu') ))

    print('Loaded Models')

    NetCinp=torch.cat((phit,phit_query_captions),1) #NetAout[:phit.shape[0],:])
    NetCout=NetC.myforward(NetCinp)
    print('Net C')
    ACloss_fn=torch.nn.MSELoss()
    Closs=ACloss_fn(NetCout,phit_taget_captions)
    print('loss C:',Closs)
    
    nn_result = []
    
    net_target=tensor(NetCout)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    
    
    for i in range (net_target.shape[1]):  #(3900): #
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
        #out += [('recall_top' + str(k) + '_correct_composition', r)]
        out.append(str(k) + ' ---> '+ str(r*100))
        r = 0.0

    print (out)

######################Training

def trainNetB1521500():
   
    inp=datasets.Feature172KOrg().PhitTargetCaption
    
    target=datasets.Features152Org().target_phix_152

    inp=torch.tensor(inp)
    target=torch.tensor(target)
    
    hidden=1500
    l_r=0.38
    epoch=100000
    batch_size=500
    save_duration=50
    seed=100
    min_error=0.3
    model_mlp=NLR3(inp.shape[1],target.shape[1],hidden).to(device)
    model_mlp.load_state_dict(torch.load( Path1+r'/NetB152_1500.pth', map_location=torch.device('cpu') ))
    loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-4) 
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
            loss = torch.mean(torch.abs(1-loss_fn(target_batch,netoutbatch)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss
        if (total_loss<min_error):
            break
        print('iteration:',j, 'total loss',total_loss,'avg loss', total_loss/(inp.shape[0]/batch_size))
        totallosses.append(total_loss)
        totallosses=[]
        total_loss=0

        s+=1
        if s==sweep_range:
            s=0
        if (j%save_duration==0) :
            torch.save(model_mlp.state_dict(), Path1+r'/NetB152_1500.pth') 

            
    print('Finished Training')
    torch.save(model_mlp.state_dict(), Path1+r'/Final_NetB152_1500.pth') 




#########################Semantic

def Semantic18(run_type):
    if run_type==0:        
        phix=datasets.Feature172KOrg().PhixQueryImg[:10000]
        phix_target=datasets.Feature172KOrg().PhixTargetImg[:10000]
        phit=datasets.Feature172KOrg().PhitQueryCaption[:10000]
        alltargetcaptions=datasets.Feature172KOrg().all_target_captions_text[:10000]
        allquerycaptions=datasets.Feature172KOrg().all_target_captions_text[:10000]#.all_Query_captions_text[:10000]
        phit_query_captions=datasets.Feature172KOrg().PhitQueryCaption[:10000]
        phit_taget_captions=datasets.Feature172KOrg().PhitTargetCaption[:10000]

    else:        
        phix=datasets.Features33KOrg().PhixQueryImg
        phix_target=datasets.Features33KOrg().PhixTargetImg
        phit=datasets.Features33KOrg().PhitQueryMod
        alltargetcaptions=datasets.Features33KOrg().all_target_captions_text
        allquerycaptions=datasets.Features33KOrg().all_queries_captions_text
        phit_query_captions=datasets.Features33KOrg().PhitQueryCaption
        phit_taget_captions=datasets.Features33KOrg().PhitTargetCaption

    
    phix=torch.tensor(phix)
    phix_target=torch.tensor(phix_target)
    hidden=1000
    NetA=NLR3T(phix.shape[1],phit.shape[1],hidden)
    NetA.load_state_dict(torch.load( Path1+r'/NetA.pth', map_location=torch.device('cpu') ))
    hidden=1000
    NetB=NLR3T(phit.shape[1],phix.shape[1],hidden)
    NetB.load_state_dict(torch.load( Path1+r'/NetB.pth', map_location=torch.device('cpu') ))
    
    hidden=1800
    NetC=NLR3S(phit.shape[1]*2,phit.shape[1],hidden)
    NetC.load_state_dict(torch.load( Path1+r'/NetCfinalUpTR.pth', map_location=torch.device('cpu') ))
    
    print('Loaded Models')
    phix=torch.tensor(phix)
    NetAout=NetA.myforward(phix)
    print('Net A')
    
    phit=torch.tensor(phit)
    phit_query_captions=torch.tensor(phit_query_captions)
    NetCinp=torch.cat((phit,NetAout[:phit.shape[0],:]),1)
    NetCout=NetC.myforward(NetCinp)
    print('Net C')
    phit_taget_captions=torch.tensor(phit_taget_captions)
    net_target=NetB.myforward(NetCout)
    print('Net A')

    ACloss_fn=torch.nn.MSELoss()
    Bloss_fn=torch.nn.CosineSimilarity()
    phit_query_captions=torch.tensor(phit_query_captions)
    Aloss=ACloss_fn(NetAout[:phit_query_captions.shape[0],:],phit_query_captions)
    phit_taget_captions=torch.tensor(phit_taget_captions)
    Closs=ACloss_fn(NetCout,phit_taget_captions)
    phix_target=torch.tensor(phix_target)
    Bloss=torch.mean(Bloss_fn(net_target,phix_target[:net_target.shape[0],:]))
    print('loss A  C  B',Aloss,Closs,Bloss)

    
    nn_result = []
    
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)
    for i in range(net_target.shape[0]):
        phix[i,:]=phix[i,:]/np.linalg.norm(phix[i,:])
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    
    
    for i in range (net_target.shape[1]):  #(3900): #
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
        #out += [('recall_top' + str(k) + '_correct_composition', r)]
        out.append(str(k) + ' ---> '+ str(r*100))
        r = 0.0

    print (out)
    
def Semantic50(run_type):
    device = torch.device("cpu")
    

    if run_type=='train': 
        PhixQueryImg =datasets.Features152Org().phix_152[:10000]
        PhitQueryCaption =datasets.Feature172KOrg().PhitQueryCaption[:10000]
        PhitQueryMod =datasets.Feature172KOrg().PhitQueryMod[:10000]
        PhixTargetImg =datasets.Features152Org().target_phix_152[:10000]
        PhitTargetCaption =datasets.Feature172KOrg().PhitTargetCaption[:10000]
        all_captions_text =datasets.Feature172KOrg().all_captions_text[:10000]
        all_target_captions_text =datasets.Feature172KOrg().all_target_captions_text[:10000]
        all_Query_captions_text =datasets.Feature172KOrg().all_Query_captions_text[:10000]
        all_ids =datasets.Feature172KOrg().all_ids[:10000]
        SearchedFeatures=PhixTargetImg
        

    elif run_type=='test':    
        PhixQueryImg=datasets.Features152Org().phix_152_test
        PhitQueryCaption=datasets.Features33KOrg().PhitQueryCaption
        PhitQueryMod=datasets.Features33KOrg().PhitQueryMod
        PhixTargetImg=datasets.Features152Org().target_phix_152_test
        PhitTargetCaption=datasets.Features33KOrg().PhitTargetCaption
        PhixAllImages=datasets.Features33KOrg().PhixAllImages
        PhitAllImagesCaptions=datasets.Features33KOrg().PhitAllImagesCaptions
        all_captions_text=datasets.Features33KOrg().all_captions_text
        all_target_captions_text=datasets.Features33KOrg().all_target_captions_text
        all_Query_captions_text=datasets.Features33KOrg().all_queries_captions_text
        all_queries_Mod_text=datasets.Features33KOrg().all_queries_Mod_text
        all_ids=datasets.Features33KOrg().all_ids
        SearchedFeatures=PhixAllImages

    
    PhitQueryCaption=torch.tensor(PhitQueryCaption).to(device)
    PhixQueryImg=torch.tensor(PhixQueryImg).to(device)
    PhitQueryMod=torch.tensor(PhitQueryMod).to(device)
    PhitTargetCaption=torch.tensor(PhitTargetCaption).to(device)
    PhixTargetImg=torch.tensor(PhixTargetImg).to(device)

    for i in range(PhixQueryImg.shape[0]):
        PhixQueryImg[i,:]/= torch.norm(PhixQueryImg[i,:])


    hidden=1000
    NetA=NLR3S(PhixQueryImg.shape[1],PhitQueryMod.shape[1],hidden).to(device)
    NetA.load_state_dict(torch.load( Path1+r'/NetA152.pth', map_location=torch.device('cpu') ))
    NetAout=NetA.myforward(PhixQueryImg)
    ACloss_fn=torch.nn.MSELoss()
    Aloss=ACloss_fn(NetAout[:PhitQueryCaption.shape[0],:],PhitQueryCaption)
    print('Net A Loss:',Aloss)


    hidden=1800
    NetC=NLR3S(PhitQueryMod.shape[1]*2,PhitQueryMod.shape[1],hidden)
    NetC.load_state_dict(torch.load( Path1+r'/NetCfinalUpTR.pth', map_location=torch.device('cpu') ))
    NetCinp=torch.cat((PhitQueryMod,NetAout[:PhitQueryMod.shape[0],:]),1)
    NetCout=NetC.myforward(NetCinp)
    Closs=ACloss_fn(NetCout,PhitTargetCaption)
    print('Net C Loss:',Closs)

    
    
    
    hidden=1000
    NetB=NLR3T(PhitTargetCaption.shape[1],PhixTargetImg.shape[1],hidden)
    NetB.load_state_dict(torch.load( Path1+r'/NetB152.pth', map_location=torch.device('cpu') ))
    net_target=NetB.myforward(NetCout)    
    Bloss_fn=torch.nn.CosineSimilarity()    
    Bloss=torch.mean(Bloss_fn(net_target,PhixTargetImg[:net_target.shape[0],:]))
    print('Net B Loss:',Bloss)


   # print(test_retrieval.testSemantic(all_captions_text,all_target_captions_text,net_target,SearchedFeatures))

    
    nn_result = []
    
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)

    for i in range(net_target.shape[0]):
        PhixTargetImg[i,:]=PhixTargetImg[i,:]/np.linalg.norm(PhixTargetImg[i,:])
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    
    
    for i in range (net_target.shape[0]):  #(3900): #
        sims = net_target[i, :].dot(PhixTargetImg[:net_target.shape[0],:].T)        
        nn_result.append(np.argsort(-sims[ :])[:110])
    
        
    
    out = []
    nn_result = [[all_target_captions_text[nn] for nn in nns] for nns in nn_result]
    
    
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
   
def Semantic152(run_type):
    device = torch.device("cpu")
    

    if run_type=='train': 
        PhixQueryImg =datasets.Features152Org().phix_152[:10000]
        PhitQueryCaption =datasets.Feature172KOrg().PhitQueryCaption[:10000]
        PhitQueryMod =datasets.Feature172KOrg().PhitQueryMod[:10000]
        PhixTargetImg =datasets.Features152Org().target_phix_152[:10000]
        PhitTargetCaption =datasets.Feature172KOrg().PhitTargetCaption[:10000]
        all_captions_text =datasets.Feature172KOrg().all_captions_text[:10000]
        all_target_captions_text =datasets.Feature172KOrg().all_target_captions_text[:10000]
        all_Query_captions_text =datasets.Feature172KOrg().all_Query_captions_text[:10000]
        all_ids =datasets.Feature172KOrg().all_ids[:10000]
        SearchedFeatures=PhixTargetImg
        

    elif run_type=='test':    
        PhixQueryImg=datasets.Features152Org().phix_152_test
        PhitQueryCaption=datasets.Features33KOrg().PhitQueryCaption
        PhitQueryMod=datasets.Features33KOrg().PhitQueryMod
        PhixTargetImg=datasets.Features152Org().target_phix_152_test
        PhitTargetCaption=datasets.Features33KOrg().PhitTargetCaption
        PhixAllImages=datasets.Features33KOrg().PhixAllImages
        PhitAllImagesCaptions=datasets.Features33KOrg().PhitAllImagesCaptions
        all_captions_text=datasets.Features33KOrg().all_captions_text
        all_target_captions_text=datasets.Features33KOrg().all_target_captions_text
        all_Query_captions_text=datasets.Features33KOrg().all_queries_captions_text
        all_queries_Mod_text=datasets.Features33KOrg().all_queries_Mod_text
        all_ids=datasets.Features33KOrg().all_ids
        SearchedFeatures=PhixAllImages

    
    PhitQueryCaption=torch.tensor(PhitQueryCaption).to(device)
    PhixQueryImg=torch.tensor(PhixQueryImg).to(device)
    PhitQueryMod=torch.tensor(PhitQueryMod).to(device)
    PhitTargetCaption=torch.tensor(PhitTargetCaption).to(device)
    PhixTargetImg=torch.tensor(PhixTargetImg).to(device)

    for i in range(PhixQueryImg.shape[0]):
        PhixQueryImg[i,:]/= torch.norm(PhixQueryImg[i,:])


    hidden=1000
    NetA=NLR3S(PhixQueryImg.shape[1],PhitQueryMod.shape[1],hidden).to(device)
    NetA.load_state_dict(torch.load( Path1+r'/NetA152.pth', map_location=torch.device('cpu') ))
    NetAout=NetA.myforward(PhixQueryImg)
    ACloss_fn=torch.nn.MSELoss()
    Aloss=ACloss_fn(NetAout[:PhitQueryCaption.shape[0],:],PhitQueryCaption)
    print('Net A Loss:',Aloss)


    hidden=1800
    NetC=NLR3S(PhitQueryMod.shape[1]*2,PhitQueryMod.shape[1],hidden)
    NetC.load_state_dict(torch.load( Path1+r'/NetCfinalUpTR.pth', map_location=torch.device('cpu') ))
    NetCinp=torch.cat((PhitQueryMod,NetAout[:PhitQueryMod.shape[0],:]),1)
    NetCout=NetC.myforward(NetCinp)
    Closs=ACloss_fn(NetCout,PhitTargetCaption)
    print('Net C Loss:',Closs)

    
    
    
    hidden=1000
    NetB=NLR3T(PhitTargetCaption.shape[1],PhixTargetImg.shape[1],hidden)
    NetB.load_state_dict(torch.load( Path1+r'/NetB152.pth', map_location=torch.device('cpu') ))
    net_target=NetB.myforward(NetCout)    
    Bloss_fn=torch.nn.CosineSimilarity()    
    Bloss=torch.mean(Bloss_fn(net_target,PhixTargetImg[:net_target.shape[0],:]))
    print('Net B Loss:',Bloss)


   # print(test_retrieval.testSemantic(all_captions_text,all_target_captions_text,net_target,SearchedFeatures))

    
    nn_result = []
    
    net_target=tensor(net_target)
    net_target=Variable(net_target,requires_grad=False)
    net_target=np.array(net_target)

    for i in range(net_target.shape[0]):
        PhixTargetImg[i,:]=PhixTargetImg[i,:]/np.linalg.norm(PhixTargetImg[i,:])
        net_target[i,:]=net_target[i,:]/np.linalg.norm(net_target[i,:])
    
    
    for i in range (net_target.shape[0]):  #(3900): #
        sims = net_target[i, :].dot(PhixTargetImg[:net_target.shape[0],:].T)        
        nn_result.append(np.argsort(-sims[ :])[:110])
    
        
    
    out = []
    nn_result = [[all_target_captions_text[nn] for nn in nns] for nns in nn_result]
    
    
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


    




# train  As PaPer:  ['1 ---> 33.21', '5 ---> 60.67', '10 ---> 73.18', '50 ---> 92.97999999999999', '100 ---> 97.19']
# test  As PaPer:  ['1 ---> 14.04121863799283', '5 ---> 34.268219832735966', '10 ---> 42.41338112305854', '50 ---> 65.31660692951016', '100 ---> 73.66487455197132']


if __name__ == '__main__': 
    trainNetB1521500()
    
    







