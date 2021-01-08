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
    for child in node.list_of_childes:
        if child.entropy == 0:
            continue
        best_node = get_best_node(child.dataset)
        if best_node is not None:
            child.list_of_childes.append(best_node)
            create_tree(child.list_of_childes[0])
        else:
            return


def get_tree_size_loop(root):
    counter = 0
    for child in root.list_of_childes:
        if child.entropy != 0:
            counter = counter + 1
            if len(child.list_of_childes) == 0:
                continue
            counter = counter + get_tree_size(child.list_of_childes[0])
    return counter


def get_tree_size(root):
    return 1 + get_tree_size_loop(root)


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
    if index == -1:
        return None
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


def test(tree, test_set):
    counter = 0
    for i in range(0, len(test_set)):
        row = test_set.iloc[i]
        predicate = do_test(tree, row)
        if predicate == (row['output']):
            counter = counter + 1
    return (counter / len(test_set)) * 100


def do_test(tree, row):
    value_in_row = row[tree.name]
    for child in tree.list_of_childes:
        if child.name.__eq__(value_in_row):
            if child.entropy == 0:
                return child.dataset['output'].iloc[0]
            else:
                if len(child.list_of_childes) == 0:
                    count = tree.dataset['output'].value_counts()
                    if count['republican'] > count['democrat']:
                        return 'republican'
                    elif count['republican'] < count['democrat']:
                        return 'democrat'
                    else:
                        random_index = numpy.random.randint(0, 2)
                        if random_index == 0:
                            return 'republican'
                        else:
                            return 'democrat'
                else:
                    return do_test(child.list_of_childes[0], row)


def main():
    names = ['output']
    for i in range(1, 17):
        names.append("v" + str(i))
    dataset = pandas.read_csv("house-votes-84.data - Copy.csv", names=names)
    replace_missing_votes(dataset)
    start = 0.3
    max_of_accuracy = 0
    min_of_accuracy = 0
    mean_of_accuracy = 0
    sum_of_accuracy = 0
    max_set = 0
    min_set = 0
    max_size = 0
    min_size = 0
    mean_of_size = 0
    sum_of_size = 0
    for i in range(0, 5):
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        training_set = dataset[:int(len(dataset) * start)]
        testing_set = dataset[int(len(dataset) * start):]
        testing_set.reset_index(drop=True, inplace=True)
        root = get_best_node(training_set)
        create_tree(root)
        tree_size = get_tree_size(root)
        sum_of_size = sum_of_size + tree_size
        print(f"tree size {tree_size}")
        success_ratio = test(root, testing_set)
        if i == 0:
            min_of_accuracy = success_ratio
            min_set = int(len(dataset) * start)
            min_size = tree_size
        sum_of_accuracy = sum_of_accuracy + success_ratio
        if success_ratio > max_of_accuracy:
            max_of_accuracy = success_ratio
            max_set = int(len(dataset) * start)
        if success_ratio < min_of_accuracy:
            min_of_accuracy = success_ratio
            min_set = int(len(dataset) * start)
        if max_size < tree_size:
            max_size = tree_size
        if min_size > tree_size:
            min_size = tree_size
        print(f"success ratio {success_ratio}%")
        start = start + 0.1
    mean_of_accuracy = sum_of_accuracy / 5
    mean_of_size = sum_of_size / 5
    print(f"max accuracy {max_of_accuracy}% and training set size {max_set}")
    print(f"min accuracy {min_of_accuracy}% and training set size {min_set}")
    print(f"mean accuracy {mean_of_accuracy}%")
    print(f"max tree size = {max_size}")
    print(f"min tree size = {min_size}")
    print(f"mean of tree size = {mean_of_size}")


main()
