import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import time
from scipy.optimize import curve_fit
import torch
import torchvision
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import PolynomialLR
import time

PATH = './models/chess_model_v2'
LOGGER_PATH = './logs/train.log'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class Logger():
    def __init__(self, file_path):
        self.file_path = file_path

    def log(self, line):
        with open(self.file_path, "a") as textfile:
            # print ('closed', textfile.closed)
            textfile.write(line)
            textfile.write('\n')

class Net(nn.Module):
    def __init__(self, input_params, output_params):
        super().__init__()
        global_params = 31
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential()
        self.layer_stack.append(nn.Linear(input_params, global_params))
        self.layer_stack.append(nn.Tanh())
        for i in range(1, 20):
            self.layer_stack.append(nn.Linear(global_params, global_params))
            self.layer_stack.append(nn.Tanh())

        self.layer_stack.append(nn.Linear(global_params, output_params))

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits


class Data(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
    # need to convert float64 to float32 else
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32))
    self.len = self.X.shape[0]
    print('len 2222', self.len)
    print('shape X', self.X.shape)
    print('shape Y', self.y.shape)
  def __getitem__(self, index: int) -> tuple:
    return self.X[index], self.y[index]
  def __len__(self) -> int:
    return self.len

class NeuralNetwork():

    model = None
    def __init__(self):
        self.logger = Logger(LOGGER_PATH)

    def predict(self, X_arg):
        self.model = torch.load(PATH)
        self.model.eval()
        with torch.no_grad():  # Wyłączenie obliczeń gradientów

            X = X_arg

            # tensor = torch.from_numpy(np.array(x))
            data = torch.tensor(X, dtype=torch.float32)

            # print('model', self.model)
            # print('model T 1', x)
            # print('model T 1', data)
            # print('model T 2', tensor)

            pred = self.model(data)
        # print('model T', pred)
        return pred
    

    def train(self, X, Y):
        self.logger.log('========================== START TRAIN ========================== X_LEN='+str(len(X)))
        XListLen = len(X)
        XList = X
        YList = Y
        start_time = time.time()

        X = X
        Y = Y
        # print(f"X: ")
        # print(XList)
        # print(XList[10])
        # X = (XList - np.min(XList)) / (np.max(XList) - np.min(XList))
        # print(f"X - NORM: ")
        # print(XList)
        # Y = (YList - np.min(YList)) / (np.max(YList) - np.min(YList))
        # def_learn_rate = self.learning_rate
        # while loss > 0.1:
        # YList = Y
        # print(f"y: ")
        print(X[0])
        print(Y[0])
        print(Y[20])
        # print(YList)
        batch_size = 1
        size_x = 64
        size_y = 66

        predictedX = X.reshape(-1, size_x)
        predictedY = Y.reshape(-1, size_y)
        data = Data(predictedX, predictedY)
        train_dataloader = DataLoader(data, batch_size=batch_size)

        try :
            self.model = torch.load(PATH)
        except :
            print('MODEL NOT EXIST')
            self.model = Net(size_x, size_y).to(device)
        # if not self.model:

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        dtype = torch.float   
        
        print(self.model)
        print('PARAMETERS OF MODEL', get_n_params(self.model))
        print('predictedX', predictedX.shape)
        print('predictedY', predictedY.shape)
        # print('start_time', start_time)

        epochs = 5000
        # test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # scheduler = MultiStepLR(optimizer, milestones=[2600, 3600, 4400], gamma=0.1)
        scheduler = MultiStepLR(optimizer, milestones=[500, 800, 1200, 1600, 2100, 2600, 3100], gamma=0.3)
        # scheduler = MultiStepLR(optimizer, milestones=[600, 1000, 1400, 1800, 2300], gamma=2)
        # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
        # scheduler = ReduceLROnPlateau(optimizer, 'max')
        # scheduler = ExponentialLR(optimizer, gamma=0.3)
        # scheduler = PolynomialLR(optimizer, total_iters=4000, power=0.1)
        epoch = 0
        while True:
            self.model.train()
            batch = 0
            size = len(predictedX)
            for i, data in enumerate(train_dataloader):
                # print('IIIIII', i)

                # Every data instance is an input + label pair
                inputs, labels = data
                # print('labels', labels)
                # print('labels', labels[0])
                # print('inputs', inputs)
                # print('inputs', inputs[0])

                # print(f"bef train ")

                # Compute prediction error
                pred = self.model(inputs[0])
                # print(f"after train ")
                # print(f"bef loss_fn ")

                loss = loss_fn(pred, labels[0])
                # print(f"after loss_fn ")

                # Backpropagation

                # print(f"bef backward ")
                loss.backward()
                # print(f"after backward ")
                # print(f"bef step ")
                optimizer.step()
                # try:

                # except Exception as error: 
                #     print("optimizer ERR ", error)

                # print(f"after step ")
                optimizer.zero_grad()
                # print(f"after zero_grad ")
                # display statistics
            # scheduler.step()

            
            loss_val = loss / (i + 1)
            loss_val = loss_val.detach().numpy()
            if epoch % 100 == 0:
                l_lr = 0
                l_lr = scheduler.get_last_lr()

                log_msg = f'Epochs:{epoch + 1:5d} |  ' \
                    f'last_LR={l_lr} | ' \
                    f'Batches per epoch: {i + 1:3d} | ' \
                    f'Loss: {loss_val:.10f}'
                
                print(log_msg)
                self.logger.log(log_msg)

            loss_msg = f'Loss: {loss / (i + 1):.10f}'
            if loss_val < 0.00001:
                print(
                    f'LEARN ENDED  ' \
                    f'Epochs:{epoch + 1:5d} |  ' \
                    f'last_LR={l_lr} | ' \
                    f'Batches per epoch: {i + 1:3d} | ' + loss_msg)
                break
            epoch = epoch + 1

            
        print(f'Epochs:{epoch + 1:5d} | ' \
        f'Batches per epoch: {i + 1:3d} | ' \
        f'Loss: {loss / (i + 1):.10f}')
        torch.save(self.model, PATH)
        exe_time = (time.time() - start_time)
        self.logger.log('||||||||||||||||||||||||||||||||| END TRAIN ||| all_epochs='+str(epoch) + ' loss='+loss_msg + ' exe_time=' + str(exe_time) + ' sec')



  
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
                
# if epoch % 20 == 0:

#     # predX = np.linspace(16, 365, (365-16)*4)
#     predX = np.linspace(start, end, 150)
#     predY = np.array([ predict(model,pX) for pX in predX])

#     ax.clear()
#     ax.plot(predX, predY)
#     ax.plot(X , Y)
#     plt.pause(0.001)

# if sys.argv[1] == 'test':
#     x = data.X
#     y = data.Y

# X = np.array([x])
# Y = np.array([y])
# print('X arg ', len(X) )
# print('y arg ', len(Y) )

# nn = Net()

# # Przykładowe dane treningowe
# X = np.random.randn(1, input_size)
# y = np.random.randn(1, output_size)

# # Trenowanie sieci
# nn.train(X, y)

