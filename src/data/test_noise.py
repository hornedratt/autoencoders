import pandas as pd
import numpy as np
import progressbar as pb
import click
import pickle
import joblib
import os
import sqlite3

from src.data.CustomDataSet import CustomDataSet

def save_to_sqlite(df, conn):
    df.to_sql('large_table', conn, if_exists='append', index=False)

def unwrapped_test_noise(amount_additional_profiles: int,
                         original_profile: pd.DataFrame,
                         noise_factor: float,
                         columns_for_profile,
                         conn
                         ) -> None:
    main = original_profile
    num = main.drop('group').drop('ID').to_numpy(dtype=float)
    final_tmp = np.array([num])
    for j in pb.progressbar(range(amount_additional_profiles)):
        tmp = num.copy()
        tmp = tmp + np.random.normal(loc=0,
                                     scale=noise_factor * tmp,
                                     size=(len(tmp)))
        tmp = abs(tmp)
        np.place(tmp, tmp > 1, 1)
        final_tmp = np.append(final_tmp, np.array([tmp]), axis=0)
    final_tmp_pd = pd.DataFrame(final_tmp)
    final_tmp_pd['group'] = main['group']
    final_tmp_pd['ID'] = main['ID']
    final_tmp_pd.columns = columns_for_profile
    save_to_sqlite(final_tmp_pd, conn)

    return None

# @click.command()
# @click.argument("input_path", type=click.Path())
# @click.argument("output_path", type=click.Path())
# @click.option("--noise", default=40, type=int)
# @click.option("--amount-additional-profiles", default=10, type=int)
def test_noise(input_path: str,
                    output_path: str,
                    noise: int = 40,
                    amount_additional_profiles: int = 800):
    """Делает зашумленный testset, таким же образом как и в trainloop: генирируем 12000(размерность
    профилей) слуйчайных величин из нормального распределения с нулевым средним и дисперсией, как в
    рассматриваемом векторе и этот ветор, домноженный на необходимый процент шума, прибавим к рассматриваемому
    вектору
    :param input_path: путь до папки, в которой лежаит csv с оригинальным набором профилей
    :param output_path: путь, куда сохраним сгенирированный csv
    :param noise: константа на которую домножаем сгенерированный вектор
    :param amount_additional_profiles: сколько зашумленных векторов сделаем из каждого оригинального
    :return: None
    """

    #   считали как DataFrame чтобы имена колонок
    original_profiles = pd.read_csv(input_path, sep=';')
    noise_factor = noise / 100
    final = pd.DataFrame({k: pd.Series(dtype=float) for k in original_profiles.columns})

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS large_table ({} TEXT, {} TEXT, {})'''.format('group', 'ID', ', '.join([f'{i} DECIMAL(11, 10)' for i in original_profiles.columns[:len(original_profiles.columns)-3]])))
    conn.commit()

    joblib.Parallel(n_jobs=5, backend='multiprocessing')(
             joblib.delayed(unwrapped_test_noise)(amount_additional_profiles,
                                                  original_profiles.loc[i].T,
                                                  noise_factor,
                                                  original_profiles.columns,
                                                  conn) for i in pb.progressbar(range(len(original_profiles))))
    return None



# if __name__ == "__main__":
#     test_noise()

if __name__ == "__main__":
    test_noise(os.path.join("..", "..", "data\\processed\\original_MS_profiles.csv"),
               os.path.join("..", "..", "data\\processed\\sets\\big_test_set_normal_noise_40%.db"))
