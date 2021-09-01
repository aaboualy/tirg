# Copyright 2019 Google Inc. All Rights Reserved.
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
# +==============================================================================

"""Provides data for training and testing."""
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

#ath1=r"D:\personal\master\MyCode\files"
Path1=r"E:\MMstr" 

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


class CSSDataset(BaseDataset):
  """CSS dataset."""

  def __init__(self, path, split='train', transform=None):
    super(CSSDataset, self).__init__()

    self.img_path = path + '/images/'
    self.transform = transform
    self.split = split
    self.data = np.load(path + '/css_toy_dataset_novel2_small.dup.npy').item()
    self.mods = self.data[self.split]['mods']
    self.imgs = []
    for objects in self.data[self.split]['objects_img']:
      label = len(self.imgs)
      if 'labels' in self.data[self.split]:
        label = self.data[self.split]['labels'][label]
      self.imgs += [{
          'objects': objects,
          'label': label,
          'captions': [str(label)]
      }]

    self.imgid2modtarget = {}
    for i in range(len(self.imgs)):
      self.imgid2modtarget[i] = []
    for i, mod in enumerate(self.mods):
      for k in range(len(mod['from'])):
        f = mod['from'][k]
        t = mod['to'][k]
        self.imgid2modtarget[f] += [(i, t)]

    self.generate_test_queries_()

  def generate_test_queries_(self):
    test_queries = []
    for mod in self.mods:
      for i, j in zip(mod['from'], mod['to']):
        test_queries += [{
            'source_img_id': i,
            'target_caption': self.imgs[j]['captions'][0],
            'mod': {
                'str': mod['to_str']
            }
        }]
    self.test_queries = test_queries

  def get_1st_training_query(self):
    i = np.random.randint(0, len(self.mods))
    mod = self.mods[i]
    j = np.random.randint(0, len(mod['from']))
    self.last_from = mod['from'][j]
    self.last_mod = [i]
    return mod['from'][j], i, mod['to'][j]

  def get_2nd_training_query(self):
    modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
    while modid in self.last_mod:
      modid, new_to = random.choice(self.imgid2modtarget[self.last_from])
    self.last_mod += [modid]
    # mod = self.mods[modid]
    return self.last_from, modid, new_to

  def generate_random_query_target(self):
    try:
      if len(self.last_mod) < 2:
        img1id, modid, img2id = self.get_2nd_training_query()
      else:
        img1id, modid, img2id = self.get_1st_training_query()
    except:
      img1id, modid, img2id = self.get_1st_training_query()

    out = {}
    out['source_img_id'] = img1id
    out['source_img_data'] = self.get_img(img1id)
    out['target_img_id'] = img2id
    out['target_img_data'] = self.get_img(img2id)
    out['mod'] = {'id': modid, 'str': self.mods[modid]['to_str']}
    return out

  def __len__(self):
    return len(self.imgs)

  def get_all_texts(self):
    return [mod['to_str'] for mod in self.mods]

  def get_img(self, idx, raw_img=False, get_2d=False):
    """Gets CSS images."""
    def generate_2d_image(objects):
      img = np.ones((64, 64, 3))
      colortext2values = {
          'gray': [87, 87, 87],
          'red': [244, 35, 35],
          'blue': [42, 75, 215],
          'green': [29, 205, 20],
          'brown': [129, 74, 25],
          'purple': [129, 38, 192],
          'cyan': [41, 208, 208],
          'yellow': [255, 238, 51]
      }
      for obj in objects:
        s = 4.0
        if obj['size'] == 'large':
          s *= 2
        c = [0, 0, 0]
        for j in range(3):
          c[j] = 1.0 * colortext2values[obj['color']][j] / 255.0
        y = obj['pos'][0] * img.shape[0]
        x = obj['pos'][1] * img.shape[1]
        if obj['shape'] == 'rectangle':
          img[int(y - s):int(y + s), int(x - s):int(x + s), :] = c
        if obj['shape'] == 'circle':
          for y0 in range(int(y - s), int(y + s) + 1):
            x0 = x + (abs(y0 - y) - s)
            x1 = 2 * x - x0
            img[y0, int(x0):int(x1), :] = c
        if obj['shape'] == 'triangle':
          for y0 in range(int(y - s), int(y + s)):
            x0 = x + (y0 - y + s) / 2
            x1 = 2 * x - x0
            x0, x1 = min(x0, x1), max(x0, x1)
            img[y0, int(x0):int(x1), :] = c
      return img

    if self.img_path is None or get_2d:
      img = generate_2d_image(self.imgs[idx]['objects'])
    else:
      img_path = self.img_path + ('/css_%s_%06d.png' % (self.split, int(idx)))
      with open(img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img = img.convert('RGB')

    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img


class Fashion200k1(BaseDataset):
  """Fashion200k dataset."""

  def __init__(self, path, split='train', transform=None):
    super(Fashion200k1, self).__init__()

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
      with open(label_path + '/' + filename) as f:
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
    print('Fashion200k:', len(self.imgs), 'images')

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
        if c not in caption2id:
          id2caption[len(caption2id)] = c
          caption2id[c] = len(caption2id)
          caption2imgids[c] = []
        caption2imgids[c].append(i)
    self.caption2imgids = caption2imgids
    print(len(caption2imgids), 'unique cations')

    # parent captions are 1-word shorter than their children
    parent2children_captions = {}
    for c in caption2id.keys():
      for w in c.split():
        p = c.replace(w, '')
        p = p.replace('  ', ' ').strip()
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
    print('Modifiable images', num_modifiable_imgs)

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
    out['target_img_id'] = target_idx
    out['target_img_data'] = self.get_img(target_idx)
    out['target_caption'] = self.imgs[target_idx]['captions'][0]
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


class MITStates(BaseDataset):
  """MITStates dataset."""

  def __init__(self, path, split='train', transform=None):
    super(MITStates, self).__init__()
    self.path = path
    self.transform = transform
    self.split = split

    self.imgs = []
    test_nouns = [
        u'armor', u'bracelet', u'bush', u'camera', u'candy', u'castle',
        u'ceramic', u'cheese', u'clock', u'clothes', u'coffee', u'fan', u'fig',
        u'fish', u'foam', u'forest', u'fruit', u'furniture', u'garden', u'gate',
        u'glass', u'horse', u'island', u'laptop', u'lead', u'lightning',
        u'mirror', u'orange', u'paint', u'persimmon', u'plastic', u'plate',
        u'potato', u'road', u'rubber', u'sand', u'shell', u'sky', u'smoke',
        u'steel', u'stream', u'table', u'tea', u'tomato', u'vacuum', u'wax',
        u'wheel', u'window', u'wool'
    ]

    from os import listdir
    for f in listdir(path + '/images'):
      if ' ' not in f:
        continue
      adj, noun = f.split()
      if adj == 'adj':
        continue
      if split == 'train' and noun in test_nouns:
        continue
      if split == 'test' and noun not in test_nouns:
        continue

      for file_path in listdir(path + '/images/' + f):
        assert (file_path.endswith('jpg'))
        self.imgs += [{
            'file_path': path + '/images/' + f + '/' + file_path,
            'captions': [f],
            'adj': adj,
            'noun': noun
        }]

    self.caption_index_init_()
    if split == 'test':
      self.generate_test_queries_()

  def get_all_texts(self):
    texts = []
    for img in self.imgs:
      texts += img['captions']
    return texts

  def __getitem__(self, idx):
    try:
      self.saved_item
    except:
      self.saved_item = None
    if self.saved_item is None:
      while True:
        idx, target_idx1 = self.caption_index_sample_(idx)
        idx, target_idx2 = self.caption_index_sample_(idx)
        if self.imgs[target_idx1]['adj'] != self.imgs[target_idx2]['adj']:
          break
      idx, target_idx = [idx, target_idx1]
      self.saved_item = [idx, target_idx2]
    else:
      idx, target_idx = self.saved_item
      self.saved_item = None

    mod_str = self.imgs[target_idx]['adj']

    return {
        'source_img_id': idx,
        'source_img_data': self.get_img(idx),
        'source_caption': self.imgs[idx]['captions'][0],
        'target_img_id': target_idx,
        'target_img_data': self.get_img(target_idx),
        'target_caption': self.imgs[target_idx]['captions'][0],
        'mod': {
            'str': mod_str
        }
    }

  def caption_index_init_(self):
    self.caption2imgids = {}
    self.noun2adjs = {}
    for i, img in enumerate(self.imgs):
      cap = img['captions'][0]
      adj = img['adj']
      noun = img['noun']
      if cap not in self.caption2imgids.keys():
        self.caption2imgids[cap] = []
      if noun not in self.noun2adjs.keys():
        self.noun2adjs[noun] = []
      self.caption2imgids[cap].append(i)
      if adj not in self.noun2adjs[noun]:
        self.noun2adjs[noun].append(adj)
    for noun, adjs in self.noun2adjs.iteritems():
      assert len(adjs) >= 2

  def caption_index_sample_(self, idx):
    noun = self.imgs[idx]['noun']
    # adj = self.imgs[idx]['adj']
    target_adj = random.choice(self.noun2adjs[noun])
    target_caption = target_adj + ' ' + noun
    target_idx = random.choice(self.caption2imgids[target_caption])
    return idx, target_idx

  def generate_test_queries_(self):
    self.test_queries = []
    for idx, img in enumerate(self.imgs):
      adj = img['adj']
      noun = img['noun']
      for target_adj in self.noun2adjs[noun]:
        if target_adj != adj:
          mod_str = target_adj
          self.test_queries += [{
              'source_img_id': idx,
              'source_caption': adj + ' ' + noun,
              'target_caption': target_adj + ' ' + noun,
              'mod': {
                  'str': mod_str
              }
          }]
    print(len(self.test_queries), 'test queries')

  def __len__(self):
    return len(self.imgs)

  def get_img(self, idx, raw_img=False):
    img_path = self.imgs[idx]['file_path']
    with open(img_path, 'rb') as f:
      img = PIL.Image.open(f)
      img = img.convert('RGB')
    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img

class Fashion200k(BaseDataset):
  """Fashion200k dataset."""

  def __init__(self, path, split='train', transform=None):
    super(Fashion200k, self).__init__()

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

    label_files = [f for f in label_files if '._' not in f]

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
    out['modifiable'] = self.imgs[idx]['modifiable']
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


###### Get From File 
class Features172K():
  def __init__(self):
    super(Features172K, self).__init__()

  #Save Values of Features Of Train DataSet For Easy Access
  def SavetoFiles(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    for i in range(172048):#172048
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

    with open(Path+r"/"+'Features172Kall_queries.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

    with open(Path+r"/"+'Features172Kall_imgs.txt', 'wb') as fp:
      pickle.dump(all_imgs, fp)

    with open(Path+r"/"+'Features172Kall_captions.txt', 'wb') as fp:
      pickle.dump(all_captions, fp)

  def SavetoFilesold(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    for i in range(172048):#172048
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

    with open(Path+r"/"+'Features172Kall_queriesold.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

    with open(Path+r"/"+'Features172Kall_imgsold.txt', 'wb') as fp:
      pickle.dump(all_imgs, fp)

    with open(Path+r"/"+'Features172Kall_captionsold.txt', 'wb') as fp:
      pickle.dump(all_captions, fp)

  def SavetoFilesImageSource(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    for i in range(172048):#172048
      print('get images=',i,end='\r')
      item = testset[i]
      img1=testset[i]['source_img_data'].numpy()
      imgs += [img1]
    
    imgs = np.concatenate(imgs)

    with open(Path+r"/"+'Features172Kall_source.txt', 'wb') as fp:
      pickle.dump(imgs, fp)

  def SavetoFilesphixt(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    for i in range(172048):#172048
      print('get images=',i,end='\r')
      item = testset[i]
      imgs += [item['source_img_data']]
      mods += [item['mod']['str']]
      if len(imgs) >= opt.batch_size or i == 9999:
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)
        f = model.extract_img_feature(imgs).data.cpu().numpy() #.cuda()
        f2 = model.extract_text_feature(mods).data.cpu().numpy() #.cuda()
        all_queries += [f]
        all_captions += [f2]
        imgs = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) >= opt.batch_size or i == 9999:
        imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0).data.cpu().numpy() #.cuda()
        all_imgs += [imgs0]
        imgs0 = []
      #all_captions += [item['target_caption']]
      #all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

    with open(Path+r"/"+'Features172Kphix.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

    with open(Path+r"/"+'Features172Kphit.txt', 'wb') as fp:
      pickle.dump(all_captions, fp)

    with open(Path+r"/"+'Features172Kphixtarget.txt', 'wb') as fp:
      pickle.dump(all_imgs, fp)

    

   

  def Get_all_queries(self):
    with open (Path1+r"/dataset172/"+'Features172Kall_queries.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_images(self):
    with open (Path1+r"/dataset172/"+'Features172Kall_imgs.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_captions(self):
    with open (Path1+r"/dataset172/"+'Features172Kall_captions.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_queriesold(self):
    with open (Path1+r"/dataset172/"+'Features172Kall_queriesold.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_imagesold(self):
    with open (Path1+r"/dataset172/"+'Features172Kall_imgsold.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_captionsold(self):
    with open (Path1+r"/dataset172/"+'Features172Kall_captionsold.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data
 
  def Get_phix(self):
    with open (Path1+r"/dataset172/"+'Features172Kphix.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_phixtarget(self):
    with open (Path1+r"/dataset172/"+'Features172Kphixtarget.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_phit(self):
    with open (Path1+r"/dataset172/"+'Features172Kphit.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data


class Features33K():
  def __init__(self):
    super(Features33K, self).__init__()

  #Save Values of Features Of Train DataSet For Easy Access
  def SavetoFiles(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    test_queries = testset.get_test_queries()
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

    with open(Path+r"/"+'Features33Kall_queries.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

    with open(Path+r"/"+'Features33Kall_imgs.txt', 'wb') as fp:
      pickle.dump(all_imgs, fp)

    with open(Path+r"/"+'Features33Kall_captions.txt', 'wb') as fp:
      pickle.dump(all_captions, fp)
    
    with open(Path+r"/"+'Features33Kall_target_captions.txt', 'wb') as fp:
      pickle.dump(all_target_captions, fp)

  def SavetoFilesold(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    test_queries = testset.get_test_queries()
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

    with open(Path+r"/"+'Features33Kall_queriesold.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

    with open(Path+r"/"+'Features33Kall_imgsold.txt', 'wb') as fp:
      pickle.dump(all_imgs, fp)

    with open(Path+r"/"+'Features33Kall_captionsold.txt', 'wb') as fp:
      pickle.dump(all_captions, fp)
    
    with open(Path+r"/"+'Features33Kall_target_captionsold.txt', 'wb') as fp:
      pickle.dump(all_target_captions, fp)

  def SavetoFiles2(self,Path,model,testset,opt):
    model.eval()
    target = []
    all_target = []
    
    test_queries = testset.get_test_queries()
    for t in tqdm(test_queries):
      target +=[testset.get_img(t['target_id'])]
      target = torch.stack(target).float()
      target = torch.autograd.Variable(target)
      f2 = model.extract_img_feature(target).data.cpu().numpy()
      all_target += [f2]
      target=[]
    all_target = np.concatenate(all_target)

    

    with open(Path+r"/"+'Features33Kall_target.txt', 'wb') as fp:
      pickle.dump(all_target, fp)

  def SavetoFiles3(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    test_queries = testset.get_test_queries()
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)#.cuda()
        #f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        f = model.extract_img_feature(imgs).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    
    
    with open(Path+r"/"+'Features33Kallsource_imgfeature.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

  def SavetoFilesphixt(self,Path,model,testset,opt):
    model.eval()
    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    imgs0 = []
    imgs = []
    mods = []
    test_queries = testset.get_test_queries()
    for t in tqdm(test_queries):
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs)#.cuda()
        f = model.extract_img_feature(imgs).data.cpu().numpy() #.cuda()
        f2 = model.extract_text_feature(mods).data.cpu().numpy() #.cuda()
        all_queries += [f]
        all_captions += [f2]
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
    #all_captions = [img['captions'][0] for img in testset.imgs]

    with open(Path+r"/"+'Features33Kphix.txt', 'wb') as fp:
      pickle.dump(all_queries, fp)

    with open(Path+r"/"+'Features33Kphit.txt', 'wb') as fp:
      pickle.dump(all_captions, fp)

    with open(Path+r"/"+'Features33Kphixtarget.txt', 'wb') as fp:
      pickle.dump(all_imgs, fp)




  def Get_all_queries(self):
      with open (Path1+r"/dataset33/"+'Features33Kall_queries.txt', 'rb') as fp:
        data = pickle.load(fp) 
        return data

  def Get_all_images(self):
    with open (Path1+r"/dataset33/"+'Features33Kall_imgs.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_captions(self):
    with open (Path1+r"/dataset33/"+'Features33Kall_captions.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data
  
  def Get_target_captions(self):
    with open (Path1+r"/dataset33/"+'Features33Kall_target_captions.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_target(self):
      with open (Path1+r"/dataset33/"+'Features33Kall_target.txt', 'rb') as fp:
        data = pickle.load(fp) 
        return data

  def Get_all_source_imgafeatured(self):
      with open (Path1+r"/dataset33/"+'Features33Kallsource_imgfeature.txt', 'rb') as fp:
        data = pickle.load(fp) 
        return data

  def Get_all_queriesold(self):
      with open (Path1+r"/dataset33/"+'Features33Kall_queriesold.txt', 'rb') as fp:
        data = pickle.load(fp) 
        return data

  def Get_all_imagesold(self):
    with open (Path1+r"/dataset33/"+'Features33Kall_imgsold.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_all_captionsold(self):
    with open (Path1+r"/dataset33/"+'Features33Kall_captionsold.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data
  
  def Get_target_captionsold(self):
    with open (Path1+r"/dataset33/"+'Features33Kall_target_captionsold.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_phix(self):
    with open (Path1+r"/dataset33/"+'Features33Kphix.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_phixtarget(self):
    with open (Path1+r"/dataset33/"+'Features33Kphixtarget.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data

  def Get_phit(self):
    with open (Path1+r"/dataset33/"+'Features33Kphit.txt', 'rb') as fp:
      data = pickle.load(fp) 
      return data
