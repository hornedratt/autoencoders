import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from src.visualization.figure_accuracy_per_epoch import losses_plot



class customdataset(Dataset):
    def __init__(self, profile, group, name):
        self.profile = torch.FloatTensor(profile)
        self.group = group
        self.name = name

    def __len__(self):
        return int(self.profile.size(dim=0))

    def __getitem__(self, index):
        profile = self.profile[index, :]
        name = self.name[index]
        group = self.group[index]
        return {'profile': profile,
                'group': group,
                'ID': name}

    def cat(self, profile, group, name):
        self.profile = torch.cat((self.profile, profile), 0)
        self.group[len(self.group):] = group
        self.name[len(self.name):] = name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#для репитативности результатов
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

import warnings
warnings.filterwarnings('ignore')

# profiles_ID = pd.read_csv("D:\education\ML\Denoising_autoencoders_for_MALDI-TOF_MS-profiles_classification/data/processed/original_MS_profiles.csv", sep=';', header=None)
# profiles_ID
#
# t = pd.read_csv("D:/education/ML/ML_autoencoders/MS_profiles/0a2c8dcb-9e19-47a4-9f35-0d7a5728eaa7", sep=';', index_col=0, header=None)
# t = t.T
# t[15001.0] = 0
#
# # пустой словарь для удобного считывания
# MS_profiles=pd.DataFrame({k: pd.Series(dtype=float) for k in t.columns})
#
# #считываем каждый файл
# for i in profiles_ID.index:
#     s = pd.read_csv("D:/education/ML/ML_autoencoders/MS_profiles/"+profiles_ID.at[i, 0], sep=';', index_col=0, header=None)
#     s = s.T
#     s[15001.0] = profiles_ID.at[i, 1]
#     s[15002.0] = profiles_ID.at[i, 2]
#     MS_profiles = MS_profiles._append(s)
#
# #делаем красивые индексы
# MS_profiles.index = profiles_ID.index

MS_profiles = pd.read_csv("C:\education\ML\Denoising_autoencoders_for_MALDI-TOF_MS-profiles_classification/data/processed/original_MS_profiles.csv", sep=';', header=None)



class vanilla_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12001, 6000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(6000, 750),
            nn.ReLU()
        )
        self.fc = nn.Linear(750, 50)

        self.unfc = nn.Linear(50, 750)

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(750, 6000),
            # nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(6000, 12001)
        )

    def forward(self, x):
        x = self.encoder(x)
        embadding = self.fc(x)
        x = self.unfc(embadding)
        reconstruction = self.decoder(x)

        return reconstruction, embadding

MS_profiles_set = customdataset (MS_profiles.iloc[:, :12001].to_numpy(),
                                 MS_profiles.iloc[:, 12001].to_numpy(),
                                 MS_profiles.iloc[:, 12002].to_numpy())

batch_s = 1
train, value = train_test_split(MS_profiles_set, train_size=0.7, shuffle=True)#получили value выбороку
train_data = DataLoader(MS_profiles_set, batch_size=batch_s, shuffle=True)#а на train отправляем весь\
#MS_profiles_set что бы модель знала профили всех имеющихся бактерий
value_data = DataLoader(MS_profiles_set, batch_size=batch_s)

L = F.mse_loss #nn.torch.functional....
autoencoder = vanilla_autoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

from torch.cuda.memory import list_gpu_processes
n_epochs = 100
train_losses = []
val_losses = []
embaddings = torch.Tensor()
truth = torch.Tensor()
pred = torch.Tensor()
noise_factor = 0.4
count = 0

for epoch in range(n_epochs):
    autoencoder.train()
    train_losses_per_epoch = []
    for X_batch in train_data:
        noise = X_batch['profile'] + \
                torch.FloatTensor(np.random.normal(loc=0.0, \
                                                   scale=noise_factor * X_batch['profile'],
                                                   size=list(X_batch['profile'].size())))  # шумим
        noise = torch.abs(noise)
        y = torch.ones(list(X_batch['profile'].size()))
        noise = torch.where(noise < 1, noise, y)
        X_batch['profile'] = X_batch['profile'].to(device)  # чистые векторы
        # print(X_batch['profile'].size().numpy())
        noise = noise.to(device)
        # print(noise.max())
        optimizer.zero_grad()
        reconstructed, embadding = autoencoder.forward(noise)  # скармливаем шум
        loss = L(reconstructed, X_batch['profile'])  # сравниваем с читыми
        loss.backward()
        optimizer.step()
        train_losses_per_epoch.append(loss.item())

    train_losses.append(np.mean(train_losses_per_epoch))

    autoencoder.eval()
    val_losses_per_epoch = []
    with torch.no_grad():
        for i, X_batch in enumerate(value_data):
            noise = X_batch['profile'] + \
                    torch.FloatTensor(np.random.normal(loc=0.0, \
                                                       scale=noise_factor * X_batch['profile'],
                                                       size=list(X_batch['profile'].size())))  # шумим
            noise = torch.abs(noise)
            y = torch.ones(list(X_batch['profile'].size()))
            noise = torch.where(noise < 1, noise, y)
            noise = noise.to(device)
            reconstructed, embadding = autoencoder(noise)
            X_batch['profile'] = X_batch['profile'].to(device)
            loss = L(reconstructed, X_batch['profile'])
            val_losses_per_epoch.append(loss.item())
            if (epoch >= (0)):
                count = count + 1
                embadding = embadding.to('cpu')
                reconstructed = reconstructed.to('cpu')
                X_batch['profile'] = X_batch['profile'].to('cpu')
                if (i == 0) and (epoch == 0):
                    embaddings = customdataset(embadding, X_batch['group'].copy(), X_batch['ID'].copy())
                    truth = customdataset(X_batch['profile'], X_batch['group'].copy(), X_batch['ID'].copy())
                    pred = customdataset(reconstructed, X_batch['group'].copy(), X_batch['ID'].copy())
                    embadding = embadding.to(device)
                if (i != 0) or (epoch != 0):
                    embaddings.cat(embadding, X_batch['group'], X_batch['ID'])
                    pred.cat(reconstructed, X_batch['group'], X_batch['ID'])
                    truth.cat(X_batch['profile'], X_batch['group'], X_batch['ID'])
                    embadding = embadding.to(device)

    val_losses.append(np.mean(val_losses_per_epoch))

# torch.save(autoencoder.encoder, output_path_model)
train_losses = [float(arr) for arr in train_losses]
val_losses = [float(arr) for arr in val_losses]
losses_plot(train_losses=train_losses,
            valid_losses=val_losses,
            output_path='reports/figures/old_results_40%.png')
