import numpy
import pandas

column_name = ['target', 'trestbps', 'chol', 'thalach', 'oldpeak', 'age', 'sex', 'cp', 'fbs', 'restecg', 'exang',
               'slope', 'ca', 'thal']


def change_zero(dataset):
    dataset['target'].replace({0: -1}, inplace=True)


def main():
    dataset = pandas.read_csv("heart.csv")
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    change_zero(dataset)
    dataset = dataset[
        ['target', 'trestbps', 'chol', 'thalach', 'oldpeak', 'age', 'sex', 'cp', 'fbs', 'restecg', 'exang',
         'slope', 'ca', 'thal']]
    needed_dataset_size = int(len(dataset) * 0.8)
    training_set = dataset[0:needed_dataset_size]
    test_set = dataset[needed_dataset_size:]
    test_set.reset_index(drop=True, inplace=True)
    parameters = start_svm(training_set)
    counter = test(test_set, parameters)
    accuracy = (counter / len(test_set)) * 100
    print(f"accuracy {accuracy}%")


def un_scale(y, mean, my_range):
    return (y * my_range) + mean


def scale_feature(x1, mean, my_range):
    return (x1 - mean) / my_range


def start_svm(scaled_features):
    parameters = list()
    for i in range(0, len(scaled_features)):
        temp = list()
        for j in range(0, len(column_name)):
            temp.append(numpy.random.randint(0, 10))
        parameters.append(temp)
    return gradient_descent(parameters, scaled_features)


def gradient_descent(parameters, scaled_features):
    learning_rate = 0.05
    number_of_iterations = 6000
    my_lambda = 1 / number_of_iterations
    counter = 0
    scaled_features_without_y = scaled_features[scaled_features.columns[1:]]
    sum_of_cost = 0
    while True:
        for i in range(0, len(scaled_features_without_y)):
            features = list()
            row = scaled_features_without_y.iloc[i]
            for j in range(0, len(row)):
                features.append(row.iloc[j])
            function_x = hypothesis(parameters[i], features)
            value = scaled_features['target'].iloc[i] * function_x
            if value >= 1.0:
                update_weights_correctly(parameters[i], learning_rate, my_lambda)
            else:
                sum_of_cost = sum_of_cost + (1 - value)
                update_weights_not_correctly(parameters[i], features, scaled_features['target'].iloc[i],
                                             learning_rate, my_lambda)
        counter = counter + 1
        if counter == number_of_iterations:
            break
    return parameters


def update_weights_correctly(parameters, learning_rate, my_lambda):
    for i in range(0, len(parameters)):
        parameters[i] = parameters[i] - learning_rate * 2 * my_lambda * parameters[i]


def update_weights_not_correctly(parameters, features, output, learning_rate, my_lambda):
    parameters[0] = parameters[0] + learning_rate * (output - (2 * my_lambda * parameters[0]))
    for i in range(1, len(parameters)):
        parameters[i] = parameters[i] + learning_rate * (
                (output * features[i - 1]) - (2 * my_lambda * parameters[i]))


def linear_equation(parameters, scaled_features):
    sum = parameters[0]
    for i in range(0, len(scaled_features)):
        sum = sum + (parameters[i + 1] * scaled_features[i])
    return sum


def hypothesis(parameters, scaled_features):
    return linear_equation(parameters, scaled_features)


def test(scaled_features, parameters):
    scaled_features_without_y = scaled_features[scaled_features.columns[1:]]
    counter = 0
    for i in range(0, len(scaled_features_without_y)):
        features = list()
        row = scaled_features_without_y.iloc[i]
        for j in range(0, len(row)):
            features.append(row.iloc[j])
        function_x = hypothesis(parameters[i], features)
        value = scaled_features['target'].iloc[i] * function_x
        if value >= 1:
            counter += 1
    return counter


main()
