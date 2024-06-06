from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from src.data.CustomDataSet import CustomDataSet

# tSNE('data/processed/sets/set_normal_noise_40%.csv', 'reports/figures/tSNE_40%.png')
def tSNE(input_path_1: str, input_path_2, input_path_3, output_path: str) -> None:

    for i, input_path in enumerate([input_path_1, input_path_2, input_path_3]):
        with open(input_path, 'rb') as file:
            data = pickle.load(file)

        X = np.array(data.profile)
        Y = data.group

        X_embedded = TSNE(n_components=2).fit_transform(X)
        principalDf = pd.DataFrame(data = X_embedded, columns = ['principal component 1', 'principal component 2'])
        finalDf = pd.concat([principalDf, data.group], axis = 1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('TSNE-groups', fontsize = 20)
        targets = pd.unique(Y)
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(targets))]
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['group'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                       , finalDf.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50)
        lgd = ax.legend(targets, loc='upper center', bbox_to_anchor=(2, 2))
        # ax.legend(targets, loc='upper right',  bbox_to_anchor=(2, 2))
        ax.grid()


        # plt.savefig(output_path+str(i)+'.png')
        fig.savefig(output_path+str(i)+'.png', bbox_inches='tight')

    return None

tSNE('../../data/processed/original_MS_profiles.pkl', '../../data/processed/sets/set_normal_noise_40%.pkl', '../../data/processed/sets/set_normal_noise_90%.pkl','../../reports/figures/tSNE_40%')

# if __name__ == "__main__":
#     tSNE()

