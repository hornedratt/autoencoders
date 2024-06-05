import torch
import pandas as pd
import numpy as np
import pickle
import click
import os
from typing import List
import progressbar as pb

from sklearn.metrics import classification_report, confusion_matrix

from src.data.CustomDataSet import CustomDataSet

@click.command()
@click.argument("output_path_acc_group", type=click.Path())
@click.argument("output_path_acc_id", type=click.Path())
@click.argument("output_path_f1_group", type=click.Path())
@click.argument("output_path_f1_id", type=click.Path())
@click.argument("noises", type=str)
def confusion_noise(output_path_acc_group: str,
                    output_path_acc_id: str,
                    output_path_f1_group: str,
                    output_path_f1_id: str,
                    noises: str):
    """ Строит матрицы для точностей и f1-мер при классификаций по группам/штаммам профилей с разным уровнем шумма с помощью
    моделей обученных на разных уровнях шума: строки отвечают за уровень шума при обучении (Train Noise), а столюцы на уровень
    шума на входе (Input Noise).
    :param output_path_acc_group: куда сохраним матрицу с точностями при  классификации по группам
    :param output_path_acc_id: куда сохраним матрицу с точностями при  классификации по штаммам
    :param output_path_f1_group: куда сохраним матрицу с f1-мерами при  классификации по группам
    :param output_path_acc_id: куда сохраним матрицу с f1-мерами при  классификации по штаммам
    :param noises: набор уровней шума для которых у нас есть тестовые наборы профилей и модели (определяется snakefile)
    """
    noises = list(map(int, noises.split(',')))
    confusion_noise_group = pd.DataFrame(np.zeros((len(noises), len(noises))),
                                         columns=pd.MultiIndex.from_tuples(
                                             [('Input Noise', str(col) + "%") for col in noises]),
                                         index=pd.MultiIndex.from_tuples(
                                             [('Train Noise', str(col) + "%") for col in noises]))
    confusion_noise_acc_group = confusion_noise_group.copy()
    confusion_noise_f1_group = confusion_noise_group.copy()
    confusion_noise_acc_id = confusion_noise_group.copy()
    confusion_noise_f1_id = confusion_noise_group.copy()

    device = torch.device('cpu')

    for i in pb.progressbar(noises):

        with open(os.path.join(f"models\\forest_{i}%_group"), 'rb') as f:
            classifier_group = pickle.load(f)

        with open(os.path.join(f"models\\forest_{i}%_ID"), 'rb') as f:
            classifier_id = pickle.load(f)

        autoencoder = torch.load(os.path.join(f"models\\DAE_norm_noise_{i}%.pkl")).to(device)
        for j in noises:
            with open(os.path.join(f"data\\processed\\sets\\test_normal_noise_{j}%.pkl"), 'rb') as file:
                valid_set = pickle.load(file)

            valid_set.profile = autoencoder(valid_set.profile)

            # дерево для групп
            pred = classifier_group.predict(valid_set.profile.detach().numpy())
            classification_rep = classification_report(valid_set.group,
                                                          pred,
                                                          output_dict=True)
            confusion_noise_acc_group.at[('Train Noise', str(i) + '%'), ('Input Noise', str(j) + '%')]\
                                                                      = classification_rep['accuracy']
            f1 = pd.DataFrame(classification_rep).transpose()
            f1 = f1.drop(index=['accuracy', 'macro avg', 'weighted avg'], axis=0)
            confusion_noise_f1_group.at[('Train Noise', str(i) + '%'), ('Input Noise', str(j) + '%')]\
                = f1.mean()['f1-score']

            # дерево для штаммов
            pred = classifier_id.predict(valid_set.profile.detach().numpy())
            classification_rep = classification_report(valid_set.name,
                                                          pred,
                                                          output_dict=True)
            confusion_noise_acc_id.at[('Train Noise', str(i) + '%'), ('Input Noise', str(j) + '%')] \
                = classification_rep['accuracy']
            f1 = pd.DataFrame(classification_rep).transpose()
            f1 = f1.drop(index=['accuracy', 'macro avg', 'weighted avg'], axis=0)
            confusion_noise_f1_id.at[('Train Noise', str(i) + '%'), ('Input Noise', str(j) + '%')] \
                = f1.mean()['f1-score']
    confusion_noise_acc_group.to_csv(output_path_acc_group, sep=';', header=True, index=True)
    confusion_noise_acc_id.to_csv(output_path_acc_id, sep=';', header=True, index=True)
    confusion_noise_f1_group.to_csv(output_path_f1_group, sep=';', header=True, index=True)
    confusion_noise_f1_id.to_csv(output_path_f1_id, sep=';', header=True, index=True)

if __name__ == "__main__":
    confusion_noise()

# confusion_noise(os.path.join("..", "..", "reports\\cross_noise_acc_group.csv"),
#                 os.path.join("..", "..", "reports\\cross_noise_acc_ID.csv"),
#                 os.path.join("..", "..", "reports\\cross_noise_f1_group.csv"),
#                 os.path.join("..", "..", "reports\\cross_noise_f1_ID.csv"),
#                 )
