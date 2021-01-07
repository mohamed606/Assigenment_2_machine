import pandas
import numpy
import matplotlib


class Node:
    entropy = -1
    name = "attribute"
    dataset = None
    list_of_childes = []

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.list_of_childes = []


def create_tree(dataset):
    columns = dataset.columns[1:]
    list_nodes = []
    first_entropy = calculate_entropy(dataset)
    max = -1
    index = -1
    for name in columns:
        parent_node = Node(name, dataset)
        count = dataset[name].value_counts()
        parent_node.entropy = first_entropy
        for i in range(0, len(count)):
            child_node = Node(count.keys()[i], dataset[dataset.get(name) == count.keys()[i]])
            child_node.entropy = calculate_entropy(child_node.dataset)
            parent_node.list_of_childes.append(child_node)
        list_nodes.append(parent_node)
        gain = information_gain(parent_node)
        print(gain)
        if max < gain:
            max = gain
            index = len(list_nodes) - 1
    return list_nodes[index]


def replace_missing_votes(dataset):
    names = dataset.columns[1:]
    for name in names:
        count = dataset[name].value_counts()
        if count['y'] > count['n']:
            dataset[name].replace({'?': 'y'}, inplace=True)
        else:
            dataset[name].replace({'?': 'n'}, inplace=True)


def calculate_entropy(dataset):
    count = dataset['output'].value_counts()
    sum_of_count = count[0] + count[1]
    positive = count[0] / sum_of_count
    negative = count[1] / sum_of_count
    return (-positive * numpy.log2(positive)) - (negative * numpy.log2(negative))


def information_gain(dataset_parent, dataset_y, dataset_n):
    return calculate_entropy(dataset_parent) - (len(dataset_y) / len(dataset_parent)) * calculate_entropy(dataset_y) - (
            len(dataset_n) / len(dataset_parent)) * calculate_entropy(dataset_n)


def information_gain(node):
    sum = 0
    length_of_parent_dataset = len(node.dataset)
    for child_node in node.list_of_childes:
        sum = sum - ((len(child_node.dataset) / length_of_parent_dataset) * child_node.entropy)
    return node.entropy + sum


def main():
    names = ['output']
    for i in range(1, 17):
        names.append("v" + str(i))
    dataset = pandas.read_csv("house-votes-84.data - Copy.csv", names=names)
    replace_missing_votes(dataset)
    node = create_tree(dataset)



main()
