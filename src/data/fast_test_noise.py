import pandas as pd
import numpy as np
import joblib
import progressbar as pb
import pickle
from sklearn.utils import shuffle


def generate_noisy_profile(num, noise_factor, amount_additional_profiles):
    final_tmp = np.array([num])
    for _ in range(amount_additional_profiles):
        tmp = num.copy()
        tmp = tmp + np.random.normal(loc=0, scale=noise_factor * tmp, size=len(tmp))
        tmp = abs(tmp)
        np.place(tmp, tmp > 1, 1)
        final_tmp = np.append(final_tmp, np.array([tmp]), axis=0)
    return final_tmp


def test_noise(input_path: str, output_path: str, noise: int = 40, amount_additional_profiles: int = 10):
    """Generates noisy test set profiles in a similar manner to train loop.

    :param input_path: path to the folder containing the original profiles csv
    :param output_path: path to save the generated csv
    :param noise: constant to multiply the generated vector
    :param amount_additional_profiles: number of noisy vectors to generate from each original
    :return: None
    """
    original_profiles = pd.read_csv(input_path, sep=';')
    noise_factor = noise / 100

    results = joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
        joblib.delayed(generate_noisy_profile)(
            original_profiles.drop(['group', 'ID'], axis=1).iloc[i].to_numpy(dtype=float),
            noise_factor,
            amount_additional_profiles
        ) for i in range(len(original_profiles))
    )

    final = pd.DataFrame()
    for i, final_tmp in enumerate(results):
        main = original_profiles.iloc[i]
        final_tmp_pd = pd.DataFrame(final_tmp)
        final_tmp_pd['group'] = main['group']
        final_tmp_pd['ID'] = main['ID']
        final_tmp_pd.columns = original_profiles.columns
        final = pd.concat([final, final_tmp_pd], axis=0)

    final = shuffle(final)
    with open(output_path, 'wb') as file:
        pickle.dump(final, file)

# Example usage
# test_noise('path_to_input.csv', 'path_to_output.pkl')
