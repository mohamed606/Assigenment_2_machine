import numpy
import pandas


class Node:
    entropy = -1
    name = "attribute"
    dataset = None
    list_of_childes = []

    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.list_of_childes = []


def calculate_entropy(dataset):
    count = dataset['target'].value_counts()
    sum_of_count = 0
    entropy = 0
    for i in range(0, len(count)):
        sum_of_count = count.iloc[i] + sum_of_count
    for i in range(0, len(count)):
        value = count.iloc[i] / sum_of_count
        entropy = entropy - (value * numpy.log2(value))
    return entropy


def information_gain(node):
    sum = 0
    length_of_parent_dataset = len(node.dataset)
    for child_node in node.list_of_childes:
        sum = sum - ((len(child_node.dataset) / length_of_parent_dataset) * child_node.entropy)
    return node.entropy + sum


def get_best_node(dataset):
    columns = dataset.columns[:13]
    list_nodes = []
    first_entropy = calculate_entropy(dataset)
    max = -1
    index = -1
    for name in columns:
        parent_node = Node(name, dataset)
        count = dataset[name].value_counts()
        parent_node.entropy = first_entropy
        for i in range(0, len(count)):
            set = dataset[dataset.get(name) == count.keys()[i]]
            del set[name]
            child_node = Node(count.keys()[i], set)
            child_node.entropy = calculate_entropy(child_node.dataset)
            parent_node.list_of_childes.append(child_node)
        list_nodes.append(parent_node)
        gain = information_gain(parent_node)
        # if max < gain:
        #     max = gain
        #     index = len(list_nodes) - 1
        print(f"gain = {gain} and feature is {name}")
    if index == -1:
        return None
    return list_nodes[index]


def change_zero(dataset):
    dataset['target'].replace({0: -1}, inplace=True)


def main():
    dataset = pandas.read_csv("heart.csv")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    change_zero(dataset)
    get_best_node(dataset)


main()
