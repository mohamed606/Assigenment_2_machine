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


def create_tree(node):
    l = []
    for child in node.list_of_childes:
        if child.entropy == 0:
            continue
        if len(l) != 0:
            for n in l:
                del child.dataset[n]
        child.list_of_childes.append(get_best_node(child.dataset))
        name = child.list_of_childes[0].name
        l.append(name)
    for child in node.list_of_childes:
        l.extend(create_tree(child.list_of_childes[0]))
    return l


def get_best_node(dataset):
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
            set = dataset[dataset.get(name) == count.keys()[i]]
            del set[name]
            child_node = Node(count.keys()[i], set)
            child_node.entropy = calculate_entropy(child_node.dataset)
            parent_node.list_of_childes.append(child_node)
        list_nodes.append(parent_node)
        gain = information_gain(parent_node)
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
    sum_of_count = 0
    entropy = 0
    for i in range(0, len(count)):
        sum_of_count = count[i] + sum_of_count
    for i in range(0, len(count)):
        value = count[i] / sum_of_count
        entropy = entropy - (value * numpy.log2(value))
    return entropy


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
    root = get_best_node(dataset)
    create_tree(root)


main()
