import torch
import pandas as pd
import numpy as np
import click
import progressbar as pb
import os

import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.VanillaAutoencoder import VanillaAutoencoder
from src.data.CustomDataSet import CustomDataSet
from src.data.CustomDataSet import collate_fn
from src.visualization.figure_accuracy_per_epoch import losses_plot

@click.command()
@click.argument("output_path_model", type=click.Path())
@click.argument("output_path_figure", type=click.Path())
@click.option("--n_epochs", default=50, type=int)
@click.option("--lr", default=0.001, type=float)
@click.option("--noise_factor", default=40, type=float)
@click.option("--batch_size", default=1, type=int)
@click.option("--set_size", default=300, type=int)
@click.option("--train_size", default=0.7, type=float)
def train_autoencoder(output_path_model: str,
                  output_path_figure: str,
                  n_epochs: int = 50,
                  lr: float = 0.001,
                  noise_factor: float = 40,
                  batch_size: int=1,
                  set_size: int=300,
                  train_size: float=0.7,
                  L=F.mse_loss):
    """Тренировка одного denoising автоенкодера с определенным уровнем шума.
     Каждый батч состоит из оригинальных профилей и в collate_fn к нему добавляется
     шум, после чего и чистый, и зашумленный батчи отдаются модели
    :param output_path_model: путь, куда сохраним готовую модельку
    :param output_path_figure: путь, куда сохраним графики с лоссами на каждой эпохе
    :param n_epochs: количество эпох
    :param lr: learning rate для Adam optimizer
    :param noise_factor: уровень шума в профилях для тренировки
    :param batch_size: размер батча
    :param set_size: общее количество профилей, которое скормим модели
    :param train_size: часть от объема из предыдущего арга, которое пойдет на тренировку
    :param L: лосс-функция
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise_factor = noise_factor / 100
    autoencoder = VanillaAutoencoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    data_set = pd.read_csv('data\\processed\\original_MS_profiles.csv', sep=';')

    train_set_size = int(set_size * train_size / len(data_set.index))
    valid_set_size = int(set_size * (1 - train_size) / len(data_set.index))

#   делаем train/valid выборки тем, что дублируем строки из оригинального df до нужного размера
#   после, окно размером с batch_size будет скользить по полученным наборам, случайно зашумлять их,
#   и отдавать на вход модели
    train_set = pd.concat([data_set] * (train_set_size + 1), axis=0, ignore_index=True)
    valid_set = pd.concat([data_set] * (valid_set_size + 1), axis=0, ignore_index=True)

    train_set = CustomDataSet(train_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              train_set['group'],
                              train_set['ID'])

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              collate_fn=lambda batch: collate_fn(batch, noise_factor=noise_factor),
                              shuffle=True)

    valid_set = CustomDataSet(valid_set.drop('group', axis=1).drop('ID', axis=1).to_numpy(dtype=float),
                              valid_set['group'],
                              valid_set['ID'])

    valid_loader = DataLoader(valid_set,
                               batch_size=batch_size,
                               collate_fn=lambda batch: collate_fn(batch, noise_factor=noise_factor),
                               shuffle=True)
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    for epoch in pb.progressbar(range(n_epochs)):
        autoencoder.train()

        for X_batch in train_loader:
            orig_profile, noise_profile, group, id = X_batch
            orig_profile = orig_profile.to(device)
            noise_profile = noise_profile.to(device)

            optimizer.zero_grad()
            reconstructed, embadding = autoencoder.forward(noise_profile)  # скармливаем шум
            loss = L(reconstructed, orig_profile)  # сравниваем с читыми
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())

        train_losses.append(np.mean(train_losses_per_epoch))

        autoencoder.eval()
        with torch.no_grad():
            for X_batch in valid_loader:
                orig_profile, noise_profile, group, id = X_batch
                orig_profile = orig_profile.to(device)
                noise_profile = noise_profile.to(device)

                reconstructed, embadding = autoencoder(noise_profile)
                loss = L(reconstructed, orig_profile)
                val_losses_per_epoch.append(loss.item())

        val_losses.append(np.mean(val_losses_per_epoch))

    torch.save(autoencoder.encoder, output_path_model)
    train_losses = [float(arr) for arr in train_losses]
    val_losses = [float(arr) for arr in val_losses]
    losses_plot(train_losses=train_losses,
                    valid_losses=val_losses,
                    output_path=output_path_figure)

if __name__ == "__main__":
    train_autoencoder()

# train_autoencoder(output_path_model=os.path.join('..', '..', 'models', f'DAE_norm_noise_{40}%.pkl'),
#               output_path_figure=os.path.join('..', '..', 'reports', 'figures', f'DAE_norm_noise_{40}%.png'))

