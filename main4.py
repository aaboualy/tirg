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
Path1=r"C:\MMaster\Files"


#################  Support Functions Section   #################
def Reform_Training_Dataset():
  
  
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
  

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'
  trig.eval()
  test_queries = train.get_test_queries()

  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  all_captions=[]
 # compute test query features
    # use training queries to approximate training retrieval performance
  imgs0 = []
  imgs = []
  mods = []
  for i in range(len(train)):
  #for i in range(50):
    
    item = train[i]
    imgs = [item['source_img_data']]
    mods = [item['mod']['str']]
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs)
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy() #.cuda()
    all_queries += [f]
    imgs0 = [item['target_img_data']]
    imgs0 = torch.stack(imgs0).float()
    imgs0 = torch.autograd.Variable(imgs0)
    imgs0 = trig.extract_img_feature(imgs0).data.cpu().numpy() #.cuda()
    all_imgs += [imgs0]
    all_captions += [item['source_caption']]
    all_target_captions += [item['target_caption']]
  all_imgs = np.concatenate(all_imgs)
  all_queries = np.concatenate(all_queries)
  with open(Path1+r"/"+'test_queries1806172k.pkl', 'wb') as fp:
    pickle.dump(test_queries, fp)

  with open(Path1+r"/"+'all_queries1806172k.pkl', 'wb') as fp:
    pickle.dump(all_queries, fp)
  with open(Path1+r"/"+'all_imgs1806172k.pkl', 'wb') as fp:
    pickle.dump(all_imgs, fp)
  with open(Path1+r"/"+'all_captions1806172k.pkl', 'wb') as fp:
    pickle.dump(all_captions, fp)
  with open(Path1+r"/"+'all_target_captions1806172k.pkl', 'wb') as fp:
    pickle.dump(all_target_captions, fp)



  
def adapt_dataset(size_limit):
  with open(Path1+r"/"+'test_queries1806172k.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

  with open(Path1+r"/"+'all_queries1806172k.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
  with open(Path1+r"/"+'all_imgs1806172k.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   #with open(Path1+r"/"+'all_captions1806172k.pkl', 'rb') as fp:
    #all_captions=pickle.load( fp)
  with open(Path1+r"/"+'all_target_captions1806172k.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  size_limit=len(all_target_captions)

  new_all_imgs=np.zeros((size_limit,512))

  for i in range(size_limit):
    if (sum(new_all_imgs[i,:])==0):
      l=[]
      t=all_target_captions[i]
      l+=[i]
      for  j in range(i+1,size_limit):
        if (all_target_captions[i]==all_target_captions[j]):
          l+=[j]
    
      tmp1=np.zeros((1,512))
      for j in range(len(l)):
        tmp1+=all_imgs[l[j],:]
      tmp1=tmp1/len(l)
      for j in range(len(l)):
        new_all_imgs[l[j],:]=tmp1
    
  with open(Path1+r"/"+'new_all_imgs2006172k.pkl', 'wb') as fp:
    pickle.dump(new_all_imgs, fp)
  
    


      


    

  
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
   with open(Path1+r"/"+'test_queries1806172k.pkl', 'rb') as fp:
    test_queries=pickle.load( fp)

   with open(Path1+r"/"+'all_queries1806172k.pkl', 'rb') as fp:
    all_queries=pickle.load( fp)
   with open(Path1+r"/"+'new_all_imgs2006172k.pkl', 'rb') as fp:
    all_imgs=pickle.load( fp)
   #with open(Path1+r"/"+'all_captions1806172k.pkl', 'rb') as fp:
    #all_captions=pickle.load( fp)
   all_queries_original=all_queries
   with open(Path1+r"/"+'all_target_captions1806172k.pkl', 'rb') as fp:
    all_target_captions=pickle.load( fp)
  all_captions=all_target_captions
  
  if (normal_beta==1 ):
    if(create_load==0):
    #################################
      distance_before=0
      distance_after=0
      new_all_queries=np.zeros((all_queries.shape[0],all_queries.shape[1]+1))
      for i in range(all_queries.shape[0]):
        f=all_queries[i,:]
        distance_before+=np.sum(abs(all_imgs[i,:]-all_queries[i, :]))
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
      distance_after+=np.sum(abs(all_imgs[i,:]-all_queries[i, :]))
    print('before after',distance_before,distance_after)

    
    
    
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
      sims = all_queries[i, :].dot(all_imgs[:int(all_imgs.shape[0]/sz)].T)
      simso=all_queries_original[i, :].dot(all_imgs[:int(all_imgs.shape[0]/sz)].T)
    else:
      sims[0,:]=np.sum(abs(all_imgs[:int(all_imgs.shape[0]/sz),:]-all_queries[i, :]),axis=1)
      simso[0,:]=np.sum(abs(all_imgs[:int(all_imgs.shape[0]/sz),:]-all_queries_original[i, :]),axis=1)

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
  out2=[]
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      for c in range(k):
        if (all_target_captions[i] == nns[c]):
          r += 1
    r /= len(nn_result)
    out2.append(str(k) + ' ---> '+ str(r*100))   
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out.append(str(k) + ' ---> '+ str(r*100))

  print(out,'out2',out2)  
  
 
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
  def __init__(self,netin,netout,nethidden1,nethidden2):
    super().__init__()
    self.netmodel= torch.nn.Sequential(torch.nn.Linear(netin, nethidden1),torch.nn.Sigmoid(),torch.nn.Linear(nethidden1, nethidden2),torch.nn.Linear(nethidden2, netout))
  def myforward (self,inv):
    outv=self.netmodel(inv)
    return outv

def build_and_train_netMSE(hidden1,hidden2,max_iterations, min_error, all_queries,all_imgs,batch_size):
  all_queries=Variable(torch.Tensor(all_queries))
  all_imgs=variable(torch.tensor(all_imgs))
  model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)
  #model=model.cuda()
  torch.manual_seed(3)
  loss_fn = torch.nn.MSELoss()
  torch.manual_seed(3)
  #criterion = nn.CosineSimilarity() 
  criterion=nn.MSELoss()
 #loss.backward()

  optimizer=torch.optim.SGD(model.parameters(), lr=0.001)
  epoch=max_iterations

  losses=[]
  totallosses=[]
  for j in range(epoch):
    total_loss=0
    for l in range(int(all_queries.shape[0]/batch_size)):
      
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
  print ('mean square loss',loss_fn(model.myforward(all_queries),all_queries))  
  print('Finished Training')
  torch.save(model.state_dict(), Path1+r'\NLPMSEt.pth') 
  
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
    all_captions=all_target_captions
  if (normal_beta_NN==2 ):
    ######### neural Network *********************************
    model=NLR2(all_queries.shape[1],all_imgs.shape[1],hiddensize)
    #torch.load(model.state_dict(), Path1+r'\NLP2.pth')
    #model.load_state_dict(torch.load(Path1+r'\'+NLP2.pth'))
    model.load_state_dict(torch.load(Path1+r"/"+model_fn))

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

def results_temp():
  stime=time.strftime("%Y%m%d-%H%M%S")
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'w')
  test_train=1
  normal_beta=0
  set_size_divider=100
  normal_normalize=0
  create_load=0
  filename='beta1806.pkl'
  dot_eucld=0
  # 1
  print(' 1 ', file=sourceFile)
  print ('1')
  #out =test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')
  test_train=1
  normal_beta=1
  set_size_divider=100
  normal_normalize=0
  create_load=0
  filename='beta1806.pkl'
  dot_eucld=0
  

  # 2
  print(' 2', file=sourceFile)
  print('2')

  #out =test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  #print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')

  test_train=1
  normal_beta=0
  set_size_divider=100
  normal_normalize=1
  create_load=0
  filename='beta1806euc.pkl'
  dot_eucld=1
  # 1
  print(' 3 ', file=sourceFile)
  print ('3 ')
  out =test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()
  sourceFile = open(Path1+r"/"+'results'+stime+'.txt', 'a')
  test_train=1
  normal_beta=1
  set_size_divider=100
  normal_normalize=0
  create_load=0
  filename='beta1806euc.pkl'
  dot_eucld=1
  

  # 2
  print(' 4', file=sourceFile)
  print('4')

  out =test_on_saved(test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  print_results(sourceFile,out,test_train,normal_beta,create_load,filename,normal_normalize, set_size_divider, dot_eucld)
  sourceFile.close()

def ab_OtestLoaded(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()
  
  test_train=1
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if (test_train)==0:
    # compute test query features
    
    all_imgs = datasets.Features33K().Get_all_images()
    all_captions = datasets.Features33K().Get_all_captions()
    all_queries = datasets.Features33K().Get_all_queries()
    all_target_captions = datasets.Features33K().Get_target_captions()

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images()  #[:100000]
    
    all_captions = datasets.Features172K().Get_all_captions() #[:100000]
    all_queries = datasets.Features172K().Get_all_queries() #[:100000]
    all_target_captions = datasets.Features172K().Get_all_captions() #[:100000]
    

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

 
def ab_Ogetvaluesfilesaved():

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
    
    asbook1 = ab_OtestLoaded(opt, trig, dataset)
    print(name,' Loaded As PaPer: ',asbook1)

    asbook = ab_Otest(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)
     

def ab_Otest(opt, model, testset):
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

def ab_Mgetvaluesfilesaved(option):

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

    
    asbook1, model, euc_model = ab_MtestLoaded(opt, trig, dataset,option)
    print(name,' Loaded As PaPer: ',asbook1, '\n  model generated      ',model, '\n    euc model',euc_model )

     

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
    new_all_queries=mymodels(all_queries,all_imgs,all_target_captions,option,test_queries)

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images() #[:10000]
    
    all_captions = datasets.Features172K().Get_all_captions() #[:10000]
    all_queries = datasets.Features172K().Get_all_queries() #[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions() #[:10000]
    
    new_all_queries=mymodels(all_queries,all_imgs,all_target_captions,option,test_queries)

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

def abt_MtestLoaded(opt, model, testset,option):
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
    new_all_queries=mymodels(all_queries,all_imgs,all_target_captions,option,test_queries)

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images()[:10000]
    
    all_captions = datasets.Features172K().Get_all_captions()[:10000]
    all_queries = datasets.Features172K().Get_all_queries()[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions()[:10000]
    
    new_all_queries=mymodels(all_queries,all_imgs,all_target_captions,option,test_queries)

  # feature normalization
  euc_new_nn_result=[]
  for i in tqdm(range(all_queries.shape[0])):
    euc_new_sims=np.sum(abs(all_imgs-all_queries[i, :]),axis=1)
    if test_queries:
      euc_new_sims[test_queries[i]['source_img_id']]=10e10

    euc_new_nn_result.append(np.argsort(euc_new_sims)[:110])

  # compute recalls
  out = []
  out2=[]
  out3=[]
  euc_new_nn_result = [[all_captions[nn] for nn in nns] for nns in euc_new_nn_result]

  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(euc_new_nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(euc_new_nn_result)
    #out += [('recall_top' + str(k) + '_correct_composition', r)]
    out3.append(str(k) + ' ---> '+ str(r*100))
    
  return out, out2, out3

def     mymodels(all_queries,all_imgs,all_target_captions,option,test_queries):
  if (option==0):
     return all_queries
  
  if (option==1):
      new_all_queries=regression(all_queries,all_imgs,0,test_queries)
  if (option==2):
    new_all_queries=neural_model(all_queries,all_imgs,0,test_queries)
  if (option==3):
    new_all_queries=neural_model(all_queries,all_imgs,1,test_queries)
  return new_all_queries

def neural_model(all_queries,all_imgs,model_option,test_queries):
  if model_option==0:
    hidden1=900
    hidden2=800
    batch_size=200
    itr=15000
    if not test_queries:
      build_and_train_netMSE(hidden1,hidden2,itr, 0.01, all_queries,all_imgs,batch_size)
    model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)

    model.load_state_dict(torch.load(Path1+r"/"+r'\NLPMSEt.pth'))
    #torch.save(model.state_dict(), Path1+r'\NLPMSE.pth') 

    model.eval()
    all_queries=Variable(torch.Tensor(all_queries))
 
    new_all_queries=model.myforward(all_queries)
    new_all_queries = torch.tensor(new_all_queries,requires_grad=False)
    #all_queries.detach().numpy()
    new_all_queries=np.array(new_all_queries)
    
    return new_all_queries
  else:
    if model_option==1:
      hidden1=900
      hidden2=800
      batch_size=200
      itr=15000
      if not test_queries:
        build_and_train_netCOS(hidden1,hidden2,itr, 0.01, all_queries,all_imgs,batch_size)
      model=NLR2(all_queries.shape[1],all_imgs.shape[1],hidden1,hidden2)

      model.load_state_dict(torch.load(Path1+r"/"+r'\NLPCOSfinal172k.pth'))
    #torch.save(model.state_dict(), Path1+r'\NLPMSE.pth') 

      model.eval()
      all_queries=Variable(torch.Tensor(all_queries))
 
      new_all_queries=model.myforward(all_queries)
      new_all_queries = torch.tensor(new_all_queries,requires_grad=False)
    #all_queries.detach().numpy()
      new_all_queries=np.array(new_all_queries)
    
      return new_all_queries
    else:
      return all_queries
  

def regression(all_queries,all_imgs,option, test_queries):
  
  if (option==0 ):
    if (test_queries):
      with open(Path1+r"/"+'beta0.pkl', 'rb') as fp:
        beta=pickle.load( fp)
      new_all_queries=np.zeros(all_queries.shape) 
      for i in range(new_all_queries.shape[0]):
        new_all_queries[i,:]=np.matmul(all_queries[i,:],beta)
    else:
      new_all_queries=all_queries.transpose()
      X1=np.matmul(new_all_queries,all_queries)  
      X2=np.linalg.inv(X1)
      X3=np.matmul(X2,new_all_queries)  
      beta=np.matmul(X3,all_imgs)
      with open(Path1+r"/"+'beta0.pkl', 'wb') as fp:
        pickle.dump(beta, fp)
      new_all_queries=np.zeros(all_queries.shape) 
      for i in range(new_all_queries.shape[0]):
        new_all_queries[i,:]=np.matmul(all_queries[i,:],beta)
    return new_all_queries

  return all_queries
        
  
if __name__ == '__main__': 
  #with open(Path1+r"/"+'all_queries172k.pkl', 'rb') as fp:
  #  all_queries=pickle.load( fp)
  #all_queries=all_queries[:10000,:]
  #with open(Path1+r"/"+'all_imgs172k.pkl', 'rb') as fp:
  #  all_imgs=pickle.load( fp)
  #all_imgs=all_imgs[:10000,:]
  #build_and_train_net(1000,5000, 0.01, all_queries,all_imgs,1000)
 
  #with open(Path1+r"/"+'all_queries172k.pkl', 'rb') as fp:
  #  all_queries=pickle.load( fp)
  #with open(Path1+r"/"+'all_imgs172k.pkl', 'rb') as fp:
  #  all_imgs=pickle.load( fp)
  #build_and_train_net(700,1000, 50, all_queries,all_imgs,200)
  #def build_and_train_net(hiddensize,max_iterations, min_error, all_queries,all_imgs,batch_size):
  #test_on_saved_NN_CMP(test_train,normal_beta_NN,create_load,filename,normal_normalize,sz,dot_eucld,hiddensize):
  #fn='NLP3.pth'
  #test_on_saved_NN_CMP(1,2,0,'nn',0,17.2,0,1000,fn)
  #test_on_saved_NN_CMP(0,0,0,'nn',0,1,0,700)
  #Reform_Training_Dataset()
  #results_temp()
  ab_Mgetvaluesfilesaved(3)
  #adapt_dataset(1000)

    

   
