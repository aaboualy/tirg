


def testLoaded(opt, model, testset):
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

  else:
    # use training queries to approximate training retrieval performance
    all_imgs = datasets.Features172K().Get_all_images()[:10000]
    
    all_captions = datasets.Features172K().Get_all_captions()[:10000]
    all_queries = datasets.Features172K().Get_all_queries()[:10000]
    all_target_captions = datasets.Features172K().Get_all_captions()[:10000]
    

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

