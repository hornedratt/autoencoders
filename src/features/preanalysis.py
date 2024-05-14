import pickle
import numpy as np
import click
import torch
# from torch.export import

from src.features.hopkins_statistic import hopkins_statistic
from src.visualization.tSNE import tSNE
from src.data.CustomDataSet import CustomDataSet


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path_tsne", type=click.Path())
@click.argument("output_path_H", type=click.Path())
def preanalysis(input_path: str,
                output_path_tsne: str,
                output_path_H: str) -> None:

    with open(input_path, 'rb') as file:
        data = pickle.load(file)

    profiles = data.profile.numpy()

#   Считаем статистику Хопкинса для, анализируемого набора данных 20 раз и усредняем.
    l = []  # list to hold values for each call
    for i in range(20):
        H = hopkins_statistic(profiles)
        l.append(H)

    result = np.mean(l)

#   Сохраняем результат
    np.savetxt(output_path_H, result, fmt='%d')

#TODO:
# поставить в этом файлике ссылочки на README или на диплом, смотря, как будешь оформлять конечный репозиторий

#   Если полученное значение выше 1/2 - можем применить tSNE
    if result > 1/2:
        tSNE(input_path=input_path, output_path=output_path_tsne)

    return None

if __name__ == "__main__":
    preanalysis()


