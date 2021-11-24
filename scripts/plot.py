import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import path

FILE_NAME = 'heart_disease_health_indicators_BRFSS2015.csv'

if __name__ == '__main__':
    # concatenate the file path
    file_path = path.data / FILE_NAME

    # read file
    df = pd.read_csv(file_path).astype('int64')

    # draw pie chart
    plt.pie(
        x=[len(df) - df.iloc[:, 0].sum(), df.iloc[:, 0].sum()],
        labels=['Healthy', 'Sick'],
        autopct='%.0f%%',
        colors=sns.color_palette('Set2'),
    )
    plt.savefig(path.plots / 'pie_chart.png')
    plt.show()

    # draw box plot
    x, y = [], []
    for name in df.columns[1:]:
        for data in df[name]:
            x.append(data)
            y.append(name)
    sns.set_theme(style='ticks')
    plt.figure(figsize=(16, 9))
    g = sns.boxplot(x=x, y=y)
    g.set_xscale("log")
    sns.despine(left=True, bottom=True)

    # save to file and display
    plt.savefig(path.plots / 'box_plot.png')
    plt.show()
