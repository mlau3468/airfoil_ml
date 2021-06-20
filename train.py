import numpy as np
import h5py
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from neuralnet import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# load dataset
fname = 'UIUC/uiuc_dataset_re200000_ncrit9.h5'
with h5py.File(fname, 'r') as f:
   geo = f['geometry'][()]
   cl = f['cl'][()]
   cd = f['cd'][()]
   cm = f['cm'][()]

# split test and train sets
batch_size = 200
test_size = 1000

dataset = [geo, cl, cd, cm]
train_data, test_data = splitTrainTestData(dataset, test_size)

train_geo, train_cl, train_cd, train_cm = groupBatch(train_data, batch_size)
n_batch = train_geo.shape[0]

test_geo, test_cl, test_cd, test_cm = groupBatch(test_data, test_size)

#train_geo, train_cl, train_cd, train_cm = reBatch([train_geo, train_cl, train_cd, train_cm])

train_geo = torch.from_numpy(train_geo).float()
train_cl = torch.from_numpy(train_cl).float()
train_cd = torch.from_numpy(train_cd).float()
train_cm = torch.from_numpy(train_cm).float()

test_geo = torch.from_numpy(test_geo).float()
test_cl = torch.from_numpy(test_cl).float()
test_cd = torch.from_numpy(test_cd).float()
test_cm = torch.from_numpy(test_cm).float()

model = Net3()
model.to(device)

# Using Adam optimizer to optimize weights of the Neural Network
optimizer = optim.Adam(model.parameters(), lr =0.001)
nEpoch = 50
train_loss = np.zeros(nEpoch)
test_loss = np.zeros(nEpoch)


for epoch in range(nEpoch):
    print('=============================Epoch {}============================='.format(epoch+1))
    #train_geo, train_cl, train_cd, train_cm = reBatch([train_geo, train_cl, train_cd, train_cm])
    batch_mse = np.zeros(n_batch)
    for i in tqdm(range(n_batch)):
        X = train_geo[i].to(device)
        y = train_cl[i].to(device)
        model.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()

        batch_mse[i] = loss.detach().to('cpu').numpy()
    
    X = test_geo[0].to(device)
    y = test_cl[0].to(device)
    output = model(X)
    loss = F.mse_loss(output, y)
    
    test_rms = np.sqrt(loss.detach().to('cpu').numpy())
    train_rms = np.sqrt(np.mean(batch_mse))
    print('Train RMS Error: {}'.format(train_rms))
    print('Test RMS Error: {}'.format(test_rms))
    train_loss[epoch] =  train_rms #rms error
    test_loss[epoch] = test_rms #rms error 

    if epoch == 0:
        fig = plt.figure()
        plt.ion()
        plt.show()
        ax = fig.add_subplot(111)
        #line1, = ax.plot(epoch+1, train_loss[epoch], color='k')
        line2, = ax.plot(epoch+1, test_loss[epoch], color='r')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMS Error')
        #fig.canvas.draw()
        plt.draw()
        plt.pause(0.1)
    else:
        #line1.set_data(np.arange(1,epoch+2), train_loss[0:epoch+1])
        line2.set_data(np.arange(1,epoch+2), test_loss[0:epoch+1])
        ax.set_xlim(0, epoch+2)
        #ax.set_ylim(np.min(np.concatenate((train_loss,test_loss), axis=0)), np.max(np.concatenate((train_loss,test_loss), axis=0)))
        ax.set_ylim(np.min(test_loss), np.max(test_loss))
        #fig.canvas.draw()
        plt.draw()
        plt.pause(0.1)

    torch.save(model, 'model.pt')