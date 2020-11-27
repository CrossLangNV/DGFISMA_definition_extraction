import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plac


@plac.annotations(
    folder=("folder dir", "positional", None, Path),
)
def main(folder: Path,
         label: str = 'weighted avg',
         ylabel: str = 'f1-score'):
    """

    Args:
        folder:
        label: '0', '1' or 'weighted avg'
        ylabel: 'precision', 'recall' or 'f1-score'

    Returns:

    """
    l_df = []

    for f in os.listdir(folder):
        if f.startswith('eval') and f.endswith('.csv'):
            int(os.path.splitext(f)[0].rsplit('_', 1)[-1].strip('s'))
            seed = int(os.path.splitext(f)[0].rsplit('_', 1)[-1].strip('s'))

            df_i = pd.read_csv(os.path.join(folder, f),
                               sep=';', index_col=0)

            df_i['seed'] = seed

            l_df.append(df_i)

    df = pd.concat(l_df, ignore_index=True)

    fig, ax = plt.subplots()
    xlabel = 'n_train'

    for name_set, group_set in df.groupby('set'):

        group_filter = group_set[group_set['label'] == label]

        l_x = []
        l_Q1 = []
        l_median = []
        l_avg = []
        l_Q3 = []

        for xlabel_i, group_xlabel_i in group_filter.groupby(xlabel):
            l_x.append(xlabel_i)

            l_ylabel = group_xlabel_i[ylabel]
            # std = statistics.stdev(l_ylabel)

            Q1, median, Q3 = l_ylabel.quantile([0.25, .5, .75])
            mean = l_ylabel.mean()

            l_Q1.append(Q1)
            l_median.append(median)
            l_Q3.append(Q3)
            l_avg.append(mean)

        l_x = np.array(l_x)
        l_Q1 = np.array(l_Q1)
        l_median = np.array(l_median)
        l_Q3 = np.array(l_Q3)
        l_avg = np.array(l_avg)

        plt.errorbar(l_x, l_avg, yerr=np.stack([l_avg - l_Q1, l_Q3 - l_avg]), fmt='.--', label=name_set)

        plt.fill_between(l_x, l_Q1, l_Q3, alpha=.2)  # I think it uses same colour as previous line!

    plt.title(f'{folder} performance. Average surrounded by Q1 and Q3')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xscale('log')

    n_min = df['n_train'].min()
    n_max = df['n_train'].max()
    10 ** math.ceil(math.log10(n_max))

    plt.xlim([10 ** math.floor(math.log10(n_min)),
              10 ** math.ceil(math.log10(n_max))])

    plt.legend()

    plt.show()

    return


if __name__ == '__main__':

    if len(sys.argv) > 1:
        plac.call(main)
    else:

        # folder = f'fasttext_eval/autotune'
        folder = f'bert_eval_wordpiece'
        # folder = f'bert_eval'

        root = os.path.join(os.path.dirname(__file__), '..', 'user_scripts', folder)
        main(root,
             label='0',
             ylabel='precision'
             )
