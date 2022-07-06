# Composing Text and Image for Image Retrieval



This Project is based on the following Paper and Code:
  Paper:
  <br>
  **<a href="https://arxiv.org/abs/1812.07119">Composing Text and Image for Image Retrieval - An Empirical Odyssey</a>**
  <br>
  Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays
  <br>
  CVPR 2019.

  Code
  <br>
  **<a href="https://github.com/google/tirg">GitHib Code</a>**

  ```
  @inproceedings{vo2019composing,
    title={Composing Text and Image for Image Retrieval-An Empirical Odyssey},
    author={Vo, Nam and Jiang, Lu and Sun, Chen and Murphy, Kevin and Li, Li-Jia and Fei-Fei, Li and Hays, James},
    booktitle={CVPR},
    year={2019}
  }
  ```

---------------------------------------------------------------------------------------

## Setup

- torchvision
- pytorch
- numpy
- tqdm
- tensorboardX
- Anaconda
- VS Code


## Introduction
In this paper, we study the task of image retrieval, where the input query is
specified in the form of an image plus some text that describes desired
modifications to the input image.

![Problem Overview](images/intro.png)


We propose a new way to combine image and
text using TIRG function for the retrieval task. We show this outperforms
existing approaches on different datasets.

![Method](images/newpipeline.png)




## Running Models

- `main.py`: driver script to run training/testing
- `datasets.py`: Dataset classes for loading images & generate training retrieval queries
- `text_model.py`: LSTM model to extract text features
- `img_text_composition_models.py`: various image text compostion models (described in the paper)
- `torch_function.py`: contains soft triplet loss function and feature normalization function
- `test_retrieval.py`: functions to perform retrieval test and compute recall performance



### Fashion200k dataset
Download the dataset from this [external website](https://github.com/xthan/fashion-200k) Download our generated test_queries.txt from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt).

Make sure the dataset include these files:

```
<dataset_path>/labels/*.txt
<dataset_path>/women/<category>/<caption>/<id>/*.jpeg
<dataset_path>/test_queries.txt`
```

Run training & testing:

```
python main.py --dataset=fashion200k --dataset_path=./Fashion200k \
  --num_iters=160000 --model=concat --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=f200k_concat

python main.py --dataset=fashion200k --dataset_path=./Fashion200k \
  --num_iters=160000 --model=tirg --loss=batch_based_classification \
  --learning_rate_decay_frequency=50000 --comment=f200k_tirg
```


## Pretrained Models:

Our pretrained models can be downloaded below. You can find our best single model accuracy:
*The numbers are slightly different from the ones reported in the paper due to the re-implementation.*

- [CSS Model](https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_css3d.pth): 0.760
- [Fashion200k Model](https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_fashion200k.pth): 0.161
- [MITStates Model](https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_mitstates.pth): 0.132


These saved weights might not be working correctly any more with new version, please refer to https://github.com/google/tirg/issues/12


## Notes:
All log files will be saved at `./runs/<timestamp><comment>`.
Monitor with tensorboard (training loss, training retrieval performance, testing retrieval performance):

```tensorboard --logdir ./runs/ --port 8888```

Pytorch's data loader might consume a lot of memory, if that's an issue add `--loader_num_workers=0` to disable loading data in parallel.
