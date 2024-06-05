import torch
import pandas as pd
import click
import progressbar as pb
import os
import pickle
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.io as pio


from src.data.CustomDataSet import CustomDataSet


def unwrapped_func(set, seed, train_size):
    targets = set.group.to_numpy()
    train_idx, test_idx = train_test_split(list(range(len(set))),
                                           train_size=train_size,
                                           shuffle=True,
                                           stratify=targets,
                                           random_state=seed)
    classifier_group = RandomForestClassifier()

    #       тренируем очередной лес
    embaddings, group, id = set.subset(train_idx)
    with torch.no_grad():
        embaddings = embaddings.numpy()
    classifier_group.fit(embaddings, group)

    #       тестируем полученный лес
    embaddings, group, id = set.subset(test_idx)
    with torch.no_grad():
        embaddings = embaddings.numpy()
    pred_group = classifier_group.predict(embaddings)

    return accuracy_score(group, pred_group)

@click.command()
@click.argument("data_path", type=click.Path())
@click.argument("model_path", type=click.Path())
@click.argument("output_path_hist_g", type=click.Path())
@click.option("--train_size", default=0.7, type=float)
@click.option("--amount", default=100, type=int)
def cross_valid(data_path: str,
                model_path: str,
                output_path_hist_g: str,
                train_size: float=0.7,
                amount: int=10):
    """Кросс-валидация
    :param data_path: путь до сэта на котором будем делать кросс-валидацию
    :param model_path: путь до кодера,который будем использовать для получения скрытых
     состояний
    :param output_path_csv: путь куда сохраним результаты
    :param output_path_hist: путь куда сохраним графики
    :param train_size: объем выборки для тренировки
    :param amoumt: количество итераций в кросс-валидации (сколько раз тренируем новую
     модель классификации)
    """
    device = torch.device('cpu')

    with open(data_path, 'rb') as file:
        valid_set = pickle.load(file)

    autoencoder = torch.load(model_path).to(device)
    valid_set.profile = autoencoder(valid_set.profile)

    accuracies_group = [joblib.Parallel(n_jobs=5, backend='multiprocessing')(
        joblib.delayed(unwrapped_func)(set=valid_set,
                                       seed=i,
                                       train_size=train_size) for i in pb.progressbar(range(amount)))]

    df = np.array(accuracies_group)
    fig = px.histogram(df,
                       title='accuracy group',
                       marginal='box',
                       labels={'count': 'amount of trains', 'value': 'accuracy'},
                       color_discrete_sequence=['indianred']
                       )
    fig.update_layout(showlegend=False)
    img_bytes = pio.to_image(fig, format="png")
    with open(output_path_hist_g, "wb") as f:
        f.write(img_bytes)

    # return None

if __name__ == "__main__":
    cross_valid()

# if __name__ == "__main__":
#     cross_valid(os.path.join("..", "..", "data\\processed\\sets\\set_normal_noise_40%.pkl"),
#                 os.path.join("..", "..", "models\\DAE_norm_noise_40%.pkl"),
#                 os.path.join("..", "..", "reports\\figures\\cross_valid_40%_result_group.png")
#                 )
