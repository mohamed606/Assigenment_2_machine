import pandas
import numpy
import matplotlib


def replace_missing_votes(dataset):
    names = dataset.columns[1:]
    for name in names:
        count = dataset[name].value_counts()
        if count['y'] > count['n']:
            dataset[name].replace({'?': 'y'}, inplace=True)
        else:
            dataset[name].replace({'?': 'n'}, inplace=True)

def calculate_entropy():
    
def main():
    names = ['output']
    for i in range(1, 17):
        names.append("v" + str(i))
    dataset = pandas.read_csv("house-votes-84.data - Copy.csv", names=names)
    replace_missing_votes(dataset)


main()
