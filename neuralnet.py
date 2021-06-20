import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def splitTrainTestData(dataset, num_test):
    num_elem = dataset[0].shape[0]
    idx = np.arange(0,num_elem,1)
    np.random.shuffle(idx)
    idx_test = idx[0:num_test]
    idx_train = idx[num_test:]

    dataset_train = []
    dataset_test = []
    for data in dataset:
        dataset_train.append(data[idx_train])
        dataset_test.append(data[idx_test])
    return dataset_train, dataset_test
    
def groupBatch(dataset, batch_size):
    n_batch = int(np.floor(dataset[0].shape[0]/batch_size))
    new_dataset = []
    for data in dataset:
        new_data = data[0:batch_size*n_batch]
        new_data = np.reshape(new_data, (n_batch,batch_size)+new_data.shape[1:])
        new_dataset.append(new_data)
    return new_dataset

def reBatch(batch_data):
    n_batch = batch_data[0].shape[0]
    batch_size = batch_data[0].shape[1]
    idx = np.arange(0,n_batch*batch_size,1)
    np.random.shuffle(idx)
    new_batch_data = []
    for data in batch_data:
        new_data = np.reshape(data, (n_batch*batch_size,)+data.shape[2:])
        new_data = new_data[idx]
        new_data = np.reshape(new_data, (n_batch,batch_size)+new_data.shape[1:])
        new_batch_data.append(new_data)
    return new_batch_data

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.flatten(x,1,2)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        
        return x

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2000, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 250)
        self.fc4 = nn.Linear(250, 1)
    
    def forward(self, x):
        x = torch.flatten(x,1,2)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        
        return x

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=10,stride=2)
        self.fc1 = nn.Linear(992, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.flatten(x,1,2)
        #print(x.shape)
        x = torch.sigmoid(self.fc1(x))
        #x = torch.sigmoid(self.fc2(x))
        x = self.fc2(x)
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        
        return x