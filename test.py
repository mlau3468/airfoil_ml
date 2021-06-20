import UIUCAirfoilDatabase
import torch
import matplotlib.pyplot as plt
from neuralnet import *


model = torch.load('model.pt')
model.to('cpu')
model.eval()

data = UIUCAirfoilDatabase.lookupUIUCData('NACA_0021.csv', 200000,9)
#data = UIUCAirfoilDatabase.lookupUIUCData('NACA_0018.csv', 200000,9)
#data = UIUCAirfoilDatabase.lookupUIUCData('E210__(13_64%)_______________.csv', 200000,9)

pts =  data['pts']
pts = groupBatch([pts], pts.shape[0])[0][0]
cl = data['cl']

pts = torch.from_numpy(pts).float()

model.zero_grad()
output = model(pts)
output = output.detach().numpy()

err = np.sqrt(np.mean((output.flatten()-cl.flatten())**2))
print(err)

plt.plot(data['a'].flatten(), cl.flatten(), label='xfoil')
plt.plot(data['a'].flatten(), output.flatten(), label='neuralnet')
plt.xlabel('Angle of Attack')
plt.ylabel('cl')
plt.legend()
plt.show()