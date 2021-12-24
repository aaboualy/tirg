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



Path1=r"D:\personal\master\MyCode\files"


#################  Class Section   #################

class BaseDataset(torch.utils.data.Dataset):
  """Base class for a dataset."""

  def __init__(self):
    super(BaseDataset, self).__init__()
    self.imgs = []
    self.test_queries = []

  def get_loader(self,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0):
    return torch.utils.data.DataLoader(
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda i: i)

  def get_test_queries(self):
    return self.test_queries

  def get_all_texts(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    return self.generate_random_query_target()

  def generate_random_query_target(self):
    raise NotImplementedError

  def get_img(self, idx, raw_img=False):
    raise NotImplementedError

class TestFashion200k(BaseDataset):
  """Fashion200k dataset."""

  def __init__(self, path, split='train', transform=None):
    super(TestFashion200k, self).__init__()

    self.split = split
    self.transform = transform
    self.img_path = path + '/'

    # get label files for the split
    label_path = path + '/labels/'
    from os import listdir
    from os.path import isfile
    from os.path import join
    label_files = [
        f for f in listdir(label_path) if isfile(join(label_path, f))
    ]
    label_files = [f for f in label_files if split in f]

    # read image info from label files
    self.imgs = []

    def caption_post_process(s):
      return s.strip().replace('.',
                               'dotmark').replace('?', 'questionmark').replace(
                                   '&', 'andmark').replace('*', 'starmark')

    for filename in label_files:
      #print('read ' + filename)
      with open(label_path + '/' + filename , encoding='utf-8') as f:
        lines = f.readlines()
      for line in lines:
        line = line.split('	')
        img = {
            'file_path': line[0],
            'detection_score': line[1],
            'captions': [caption_post_process(line[2])],
            'split': split,
            'modifiable': False
        }
        self.imgs += [img]
    print ('Test Fashion200k:', len(self.imgs), 'images')
    global testimagedata
    testimagedata=self.imgs
    # generate query for training or testing
    if split == 'train':
      self.caption_index_init_()
    else:
      self.generate_test_queries_()

  def get_different_word(self, source_caption, target_caption):
    source_words = source_caption.split()
    target_words = target_caption.split()
    for source_word in source_words:
      if source_word not in target_words:
        break
    for target_word in target_words:
      if target_word not in source_words:
        break
    mod_str = 'replace ' + source_word + ' with ' + target_word
    return source_word, target_word, mod_str

  def generate_test_queries_(self):
    file2imgid = {}
    for i, img in enumerate(self.imgs):
      file2imgid[img['file_path']] = i
    with open(self.img_path + '/test_queries.txt') as f:
      lines = f.readlines()
    self.test_queries = []
    for line in lines:
      source_file, target_file = line.split()
      idx = file2imgid[source_file]
      target_idx = file2imgid[target_file]
      source_caption = self.imgs[idx]['captions'][0]
      target_caption = self.imgs[target_idx]['captions'][0]
      source_word, target_word, mod_str = self.get_different_word(
          source_caption, target_caption)
      self.test_queries += [{
          'source_img_id': idx,
          'source_caption': source_caption,
          'target_caption': target_caption,
          'target_id':target_idx,
          # 'source_data':self.get_img(idx),
          # 'target_data':self.get_img(target_idx),
          'mod': {
              'str': mod_str
          }
      }]

  def caption_index_init_(self):
    """ index caption to generate training query-target example on the fly later"""

    # index caption 2 caption_id and caption 2 image_ids
    caption2id = {}
    id2caption = {}
    caption2imgids = {}
    for i, img in enumerate(self.imgs):
      for c in img['captions']:
        #if not caption2id.has_key(c):
        if c not in caption2id:
          id2caption[len(caption2id)] = c
          caption2id[c] = len(caption2id)
          caption2imgids[c] = []
        caption2imgids[c].append(i)
    self.caption2imgids = caption2imgids
    print (len(caption2imgids), 'unique cations')

    # parent captions are 1-word shorter than their children
    parent2children_captions = {}
    for c in caption2id.keys():
      for w in c.split():
        p = c.replace(w, '')
        p = p.replace('  ', ' ').strip()
        #if not parent2children_captions.has_key(p):
        if p not in parent2children_captions:
          parent2children_captions[p] = []
        if c not in parent2children_captions[p]:
          parent2children_captions[p].append(c)
    self.parent2children_captions = parent2children_captions

    # identify parent captions for each image
    for img in self.imgs:
      img['modifiable'] = False
      img['parent_captions'] = []
    for p in parent2children_captions:
      if len(parent2children_captions[p]) >= 2:
        for c in parent2children_captions[p]:
          for imgid in caption2imgids[c]:
            self.imgs[imgid]['modifiable'] = True
            self.imgs[imgid]['parent_captions'] += [p]
    num_modifiable_imgs = 0
    for img in self.imgs:
      if img['modifiable']:
        num_modifiable_imgs += 1
    print ('Modifiable images', num_modifiable_imgs)

  def caption_index_sample_(self, idx):
    while not self.imgs[idx]['modifiable']:
      idx = np.random.randint(0, len(self.imgs))

    # find random target image (same parent)
    img = self.imgs[idx]
    while True:
      p = random.choice(img['parent_captions'])
      c = random.choice(self.parent2children_captions[p])
      if c not in img['captions']:
        break
    target_idx = random.choice(self.caption2imgids[c])

    # find the word difference between query and target (not in parent caption)
    source_caption = self.imgs[idx]['captions'][0]
    target_caption = self.imgs[target_idx]['captions'][0]
    source_word, target_word, mod_str = self.get_different_word(
        source_caption, target_caption)
    return idx, target_idx, source_word, target_word, mod_str

  def get_all_texts(self):
    texts = []
    for img in self.imgs:
      for c in img['captions']:
        texts.append(c)
    return texts

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
        idx)
    out = {}
    out['source_img_id'] = idx
    out['source_img_data'] = self.get_img(idx)
    out['source_caption'] = self.imgs[idx]['captions'][0]
    out['source_path'] = self.imgs[idx]['file_path']
    out['target_img_id'] = target_idx
    out['target_img_data'] = self.get_img(target_idx)
    out['target_caption'] = self.imgs[target_idx]['captions'][0]
    out['target_path'] = self.imgs[target_idx]['file_path']
    out['mod'] = {'str': mod_str}
    return out

  def get_img(self, idx, raw_img=False):
    img_path = self.img_path + self.imgs[idx]['file_path']
    with open(img_path, 'rb') as f:
      img = PIL.Image.open(f)
      img = img.convert('RGB')
    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img

class Fashion200k():
  def __init__(self, path, split='train', transform=None):
    super(Fashion200k, self).__init__()

    self.split = split
    self.transform = transform
    self.img_path = path + '\\'

    # get label files for the split
    label_path = path + '\\labels\\'
    from os import listdir
    from os.path import isfile
    from os.path import join
    label_files = [
        f for f in listdir(label_path) if isfile(join(label_path, f))
    ]
    
    label_files = [f for f in label_files if split in f]
    
    # read image info from label files
    
    self.imgs = []

    def caption_post_process(s):
      return s.strip().replace('.',
                                  'dotmark').replace('?', 'questionmark').replace(
                                      '&', 'andmark').replace('*', 'starmark')

  
    for filename in label_files:
      #print('read ' + filename)
      #i=0
      #if(filename == 'top_train_detect_all.txt'):
      with open(label_path + '/' + filename , encoding='utf-8') as f:
          lines = f.readlines()
          for line in lines:
              line = line.split('	')
              img = {
                  'file_path': line[0],
                  'detection_score': line[1],
                  'captions': [caption_post_process(line[2])],
                  'split': split,
                  'modifiable': False
              }
          
              self.imgs += [img]
              
          
    global imagedata
    imagedata=self.imgs
    print ('All Fashion200k:', len(self.imgs), 'images')
   
  def caption_index_sample_(self, idx):
    
    while not self.imgs[idx]['modifiable']:
      idx = np.random.randint(0, len(self.imgs))

    # find random target image (same parent)
    img = self.imgs[idx]
    while True:
      p = random.choice(img['parent_captions'])
      c = random.choice(self.parent2children_captions[p])
      if c not in img['captions']:
        break
    target_idx = random.choice(self.caption2imgids[c])

    # find the word difference between query and target (not in parent caption)
    source_caption = self.imgs[idx]['captions'][0]
    target_caption = self.imgs[target_idx]['captions'][0]
    source_word, target_word, mod_str = self.get_different_word(
        source_caption, target_caption)
     
    return idx, target_idx, source_word, target_word, mod_str

  def get_all_texts(self):
    texts = []
    for img in self.imgs:
      for c in img['captions']:
        texts.append(c)
    return texts

  def __getitem__(self,idx):
    out = {}
    
    out['image'] = self.get_img(self.imgs[idx]['file_path'])
    out['caption'] = self.imgs[idx]['captions'][0]
    out['index']=idx
    string1 =str(Path((self.imgs[1]['file_path'])).parents[1])
    #classes1=string1.split('\\')[-1]
    #out['label'] = classes.index(classes1)
    out['url']=self.imgs[idx]['file_path']
    
    return out

  def __len__(self):
    return len(self.imgs)
  
  def get_img(self, idx, raw_img=False):
      
      #img_path = self.img_path + self.imgs[idx]['file_path']
      img_path = self.img_path + idx
      #with open(img_path, 'rb') as f:
      #if os.path.exists(img_path):
      img = Image.open(img_path)      
      img = img.convert('RGB')      
      if raw_img:
        return img
      if self.transform:
        img = self.transform(img)
      
     
      return img
      #else:
        #return 'Delete'  

class ConCatModule(torch.nn.Module):

  def __init__(self):
    super(ConCatModule, self).__init__()

  def forward(self, x):
    x = torch.cat(x, dim=1)
    return x

class ImgTextCompositionBase(torch.nn.Module):
  """Base class for image + text composition."""

  def __init__(self):
    super(ImgTextCompositionBase, self).__init__()
    self.normalization_layer = torch_functions.NormalizationLayer(
        normalize_scale=4.0, learn_scale=True)
    self.soft_triplet_loss = torch_functions.TripletLoss()

  def extract_img_feature(self, imgs):
    raise NotImplementedError

  def extract_text_feature(self, texts):
    raise NotImplementedError

  def compose_img_text(self, imgs, texts):
    raise NotImplementedError

  def compute_loss(self,
                   imgs_query,
                   modification_texts,
                   imgs_target,
                   soft_triplet_loss=True):
    mod_img1 = self.compose_img_text(imgs_query, modification_texts)
    mod_img1 = self.normalization_layer(mod_img1)
    img2 = self.extract_img_feature(imgs_target)
    img2 = self.normalization_layer(img2)
    assert (mod_img1.shape[0] == img2.shape[0] and
            mod_img1.shape[1] == img2.shape[1])
    if soft_triplet_loss:
      return self.compute_soft_triplet_loss_(mod_img1, img2)
    else:
      return self.compute_batch_based_classification_loss_(mod_img1, img2)

  def compute_soft_triplet_loss_(self, mod_img1, img2):
    triplets = []
    labels = [*range(mod_img1.shape[0]) , *range(img2.shape[0])]
    for i in range(len(labels)):
      triplets_i = []
      for j in range(len(labels)):
        if labels[i] == labels[j] and i != j:
          for k in range(len(labels)):
            if labels[i] != labels[k]:
              triplets_i.append([i, j, k])
      np.random.shuffle(triplets_i)
      triplets += triplets_i[:3]
    assert (triplets and len(triplets) < 2000)
    return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

  def compute_batch_based_classification_loss_(self, mod_img1, img2):
    img2m=img2.transpose(0, 1)
    x = torch.mm(mod_img1,img2m )
    labels = torch.tensor(range(x.shape[0])).long()
    labels = torch.autograd.Variable(labels)
    return F.cross_entropy(x, labels)

class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
  """Base class for image and text encoder."""

  def __init__(self, texts, embed_dim):
    super(ImgEncoderTextEncoderBase, self).__init__()

    
    img_model = torchvision.models.resnet18(pretrained=False)
    img_model.load_state_dict(torch.load(Path1+r'\resnet18-5c106cde.pth'))
    
    class GlobalAvgPool2d(torch.nn.Module):
      def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))

   
    img_model.avgpool = GlobalAvgPool2d()
    img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
    self.img_model = img_model

    # text model
    self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=embed_dim,
            lstm_hidden_dim=embed_dim)

  def extract_img_feature(self, imgs):
    return self.img_model(imgs)

  def extract_text_feature(self, texts):
    return self.text_model(texts)

class SimpleModelImageOnly(ImgEncoderTextEncoderBase):

  def compose_img_text(self, imgs, texts):
    return self.extract_img_feature(imgs)

class SimpleModelTextOnly(ImgEncoderTextEncoderBase):

  def compose_img_text(self, imgs, texts):
    return self.extract_text_feature(texts)

class Concat(ImgEncoderTextEncoderBase):
  """Concatenation model."""

  def __init__(self, texts, embed_dim):
    super(Concat, self).__init__(texts, embed_dim)

    # composer
    class Composer(torch.nn.Module):
      """Inner composer class."""

      def __init__(self):
        super(Composer, self).__init__()
        self.m = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

      def forward(self, x):
        f = torch.cat(x, dim=1)
        f = self.m(f)
        return f

    self.composer = Composer()

  def compose_img_text(self, imgs, texts):
    img_features = self.extract_img_feature(imgs)
    text_features = self.extract_text_feature(texts)
    return self.compose_img_text_features(img_features, text_features)

  def compose_img_text_features(self, img_features, text_features):
    return self.composer((img_features, text_features))

class TIRG(ImgEncoderTextEncoderBase):
  """The TIGR model.
  The method is described in
  Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
  "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
  CVPR 2019. arXiv:1812.07119
  """

  def __init__(self, texts, embed_dim):
        super(TIRG, self).__init__(texts, embed_dim)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

 
  def compose_img_text(self, imgs, texts):
      img_features = self.extract_img_feature(imgs)
      text_features = self.extract_text_feature(texts)
      return self.compose_img_text_features(img_features, text_features)
      

  def compose_img_text_features(self, img_features, text_features):
       f1 = self.gated_feature_composer((img_features, text_features))
       f2 = self.res_info_composer((img_features, text_features))
       f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
       return f


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



#################  Beta Functions Section   #################


def getbeta():
  trainset1,trainloader1=dataset(2)
  
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
    
  trig= TIRG([t.encode().decode('utf-8') for t in trainset1.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()
  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]
 
  
  for Data in tqdm(testset.get_test_queries()):
    
    
    imgs += [testset.get_img(Data['source_img_id'])]
    mods += [Data['mod']['str']]
    target +=[testset.get_img(Data['target_id'])]
    
    
    imgs = torch.stack(imgs)#.float()
    imgs = torch.autograd.Variable(imgs)
    mods = [t for t in mods]
    #print(mods)
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()#.data.cpu().numpy()

    target = torch.stack(target)#.float()
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

  # trigdata1=trigdata.transpose()

  # X11=np.matmul(trigdata1,trigdata)  
  # X21=np.linalg.inv(X11)
  # X31=np.matmul(X21,trigdata1)  
  # beta=np.matmul(X31,imgdata)  


  with open(Path1+r"/"+'BetaNormalized.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

  # with open(Path1+r"/"+'BetaNotNormalized.txt', 'wb') as fp:
  #   pickle.dump(beta, fp)

def GetValues():
  
  with open (Path1+"/BetaNormalized.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 

  trainset = TestFashion200k(
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
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset),
    
    # betaN = test_retrieval.testbetaNot(opt, trig, dataset,Beta)
    # print('BetaNotNormalized: ',betaN)
    
    try:
      betaNor = test_retrieval.testbetanormalizednot(opt, trig, dataset,BetaNormalize)
      print(name,' BetaNormalized: ',betaNor)
    except:
      print('ERROR')

    try:
      asbook = test_retrieval.test(opt, trig, dataset)
      print(name,' As PaPer: ',asbook)
    except:
      print('ERROR')






def getbetatrain():
  trainset1,trainloader1=dataset(2)
  
  train = TestFashion200k(
        path=Path1,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ]))

  trainloader = torch.utils.data.DataLoader(train, batch_size=1,
                                            shuffle=False, num_workers=0)

    
  trig= TIRG([t.encode().decode('utf-8') for t in trainset1.get_all_texts()],512)
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.eval()
  imgs = []
  mods = []
  trigdata=[]
  target=[]
  imgdata=[]
 
  
  for Data in tqdm(trainloader):
    
    
    imgs += [train.get_img(Data['source_img_id'])]
    mods += [Data['mod']['str']]
    target +=[train.get_img(Data['target_img_id'])]
    
    
    imgs = torch.stack(imgs)#.float()
    imgs = torch.autograd.Variable(imgs)
    mods = [t for t in mods[0]]
    #print(mods)
    f = trig.compose_img_text(imgs, mods).data.cpu().numpy()#.data.cpu().numpy()

    target = torch.stack(target)#.float()
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

  # trigdata1=trigdata.transpose()

  # X11=np.matmul(trigdata1,trigdata)  
  # X21=np.linalg.inv(X11)
  # X31=np.matmul(X21,trigdata1)  
  # beta=np.matmul(X31,imgdata)  


  with open(Path1+r"/"+'trainBetaNormalized.txt', 'wb') as fp:
    pickle.dump(Nbeta, fp)

  # with open(Path1+r"/"+'BetaNotNormalized.txt', 'wb') as fp:
  #   pickle.dump(beta, fp)

def GetValuestrain():
  
  with open (Path1+"/trainBetaNormalized.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 

  trainset = TestFashion200k(
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
  #trig.load_state_dict(torch.load(Path1+r'\checkpoint_fashion200k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  trig.load_state_dict(torch.load(Path1+r'\fashion200k.tirg.iter160k.pth' , map_location=torch.device('cpu') )['model_state_dict'])
  

  opt = argparse.ArgumentParser()
  opt.add_argument('--batch_size', type=int, default=2)
  opt.add_argument('--dataset', type=str, default='fashion200k')
  opt.batch_size =1
  opt.dataset='fashion200k'
  
  for name, dataset in [ ('train', trainset),('test', testset)]: #('train', trainset),
    
    # betaN = test_retrieval.testbetaNot(opt, trig, dataset,Beta)
    # print('BetaNotNormalized: ',betaN)
    
    #try:
    betaNor = test_retrieval.testbetanormalizednot(opt, trig, dataset,BetaNormalize)
    print(name,' BetaNormalized: ',betaNor)
    #except:
    #  print('ERROR')

    #try:
    asbook = test_retrieval.test(opt, trig, dataset)
    print(name,' As PaPer: ',asbook)
    #except:
    #  print('ERROR')

def GetValuestrain15time():
  
  with open (Path1+"/trainBetaNormalized.txt", 'rb') as fp:
    BetaNormalize = pickle.load(fp) 

  trainset = TestFashion200k(
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
      





    
if __name__ == '__main__': 
    #getbeta() 
    #GetValues()
    getbetatrain()
    GetValuestrain()
    #GetValuestrain15time()
