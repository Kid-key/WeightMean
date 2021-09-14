import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from folder2lmdb import ImageFolderLMDB


def data_sets(root):
    #traindir = os.path.join('/data0','%s.lmdb'%'train')
    #valdir = os.path.join('/data0','%s.lmdb'%'val')
    #valdir = '/home/data/val'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform=transforms.Compose([
            transforms.RandomResizedCrop(224),  #(0.5,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    val_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    traindir = os.path.join(root,'train')
    train_dataset = datasets.ImageFolder(traindir,train_transform)
    valdir = os.path.join(root,'val')
    val_dataset = datasets.ImageFolder(valdir,val_transform)

    #train_dataset=ImageFolderLMDB(traindir,train_transform)

    #val_dataset=ImageFolderLMDB(valdir,val_transform)
    #print(train_dataset.__getitem__(1))

    return train_dataset, val_dataset
