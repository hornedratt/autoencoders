import torch
import pandas as pd
import click
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.data.CustomDataSet import CustomDataSet

@click.command()
@click.argument("model_path", type=click.Path())
@click.argument("data_path", type=click.Path())
@click.argument("output_data_test", type=click.Path())
@click.argument("output_path_group", type=click.Path())
@click.argument("output_path_group_rep", type=click.Path())
@click.argument("output_path_id", type=click.Path())
@click.argument("output_path_id_rep", type=click.Path())
def train_forest(model_path: str,
                 data_path: str,
                 output_data_test: str,
                 output_path_group: str,
                 output_path_group_rep: str,
                 output_path_id: str,
                 output_path_id_rep: str):
    """Тренировка и тест 2-х случайных лесов: один по классифицирует погруппам, другой
    по штаммам. В data_path лежит полный сэт, его test часть сохраним отдельно, для использования в cross_noise,
    в тесте вместо профилей уже лежат эммбединги
    :param model_path: путь до кодера,который будем использовать для получения скрытых
    состояний
    :param data_path: путь до сэта на котором будем тренировать случайный лес
    :param output_path_group: путь куда сохраним готовый лес, классифицирующий по группам
    :param output_path_group_rep: путь куда сохраним отчет в формате csv о тесте леса,
     классифицирующего по группам
    :param output_path_id: путь куда сохраним готовый лес, классифицирующий по штаммам
    :param output_path_id_rep: путь куда сохраним отчет в формате csv о тесте  леса,
     классифицирующего по щтаммам
    """
    device = torch.device('cpu')

    with open(data_path, 'rb') as file:
        train_set = pickle.load(file)

    size = len(train_set)
    autoencoder = torch.load(model_path).to(device)
    targets = train_set.group.to_numpy()

#   разбили на train и test
    idx_train, idx_test = train_test_split(list(range(size)), train_size=0.7, stratify=targets)
    x_train, y_train_group, y_train_id = train_set.subset(idx_train)
    x_test, y_test_group, y_test_id = train_set.subset(idx_test)

#   до прогнки через кодер, чистые профили для теста с метками сохранили для cross_noise
    test_set = CustomDataSet(x_test, y_test_group, y_test_id)
    with open(output_data_test, 'wb') as file:
        pickle.dump(test_set, file)

    x_train = autoencoder(x_train).detach().numpy()
    x_test = autoencoder(x_test).detach().numpy()

    classifier = RandomForestClassifier(n_estimators=10, min_samples_split=8, min_samples_leaf=4)
    classifier.fit(x_train, y_train_group)
    y_pred = classifier.predict(x_test)
    classification_report_group = classification_report(y_test_group,
                                                        y_pred,
                                                        output_dict=True)
    classification_report_group = pd.DataFrame(classification_report_group).T
    classification_report_group.to_csv(output_path_group_rep)
    with open(output_path_group, 'wb') as f:
        pickle.dump(classifier, f)

    classifier = RandomForestClassifier(n_estimators=10, min_samples_split=8, min_samples_leaf=4)
    classifier.fit(x_train, y_train_id)
    y_pred = classifier.predict(x_test)
    classification_report_id = classification_report(y_test_id,
                                                        y_pred,
                                                        output_dict=True)
    classification_report_id = pd.DataFrame(classification_report_id).T
    classification_report_id.to_csv(output_path_id_rep)
    with open(output_path_id, 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == "__main__":
    train_forest()

# train_forest("..\\..\\models\\DAE_norm_noise_40%.pkl",
#             "..\\..\\data\\processed\\sets\\set_normal_noise_40%.pkl",
#             "..\\..\\models\\forest_40%_group",
#             "..\\..\\reports\\forest_40%_group.csv",
#             "..\\..\\models\\forest_40%_ID",
#             "..\\..\\reports\\forest_40%_group.csv",
#             "..\\..\\reports\\forest_40%_ID.csv"
#              )