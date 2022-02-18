import matplotlib.pyplot as plt
import random
import numpy

# Get data from a file:
# Return list of data with values
def get_data(name):
    data_set = []
    file = open(name, "r")
    for line in file:
        line_split = line.split(",")
        corrected = [element.replace('\n', '') for element in line_split]
        data_set.append(corrected)
    return data_set

# Get all the total possible outcomes (Y's) and how many times were they chosen 
def get_y(data_set, parameter, data_length ):
    y_table = {}

    for data in range(data_length):
        result = data_set[data][parameter]

        if result in y_table.keys():
            y_table[result] += 1
        else:
            y_table.update({result:1})

    return y_table

# Create the probability table for the given data set
def create_table_prob(data_set, parameter):

    prob_table = []
    data_length = len(data_set)
    attributes_len = len(data_set[0])
    for attr in range(attributes_len):

        if attr == parameter:
            prob_table.append(get_y(data_set, parameter, data_length))
            continue

        attribute_table = {}
        for data in range(data_length):
            attr_key = data_set[data][attr]
            result = data_set[data][parameter]
            if attr_key  in attribute_table.keys():

                if result  in attribute_table[attr_key].keys():
                    attribute_table[attr_key][result] += 1
                else:
                    attribute_table[attr_key].update({result:1})
            else:
                attribute_table.update({attr_key:{result:1}})

        prob_table.append(attribute_table)
    return prob_table

# Predict the outcome for the given attributes using the learned data set
def make_prediction(prob_table, tested, parameter, total):

    predicted_answer = {}
    for key in prob_table[parameter]:
        result_y = prob_table[parameter][key]

        prob_of_outcome = 1
        for i in range(0,len(tested)):
            if i == parameter:
                continue
            attr_wanted = tested[i]
            if attr_wanted not in prob_table[i].keys():
                continue
            elif key not in prob_table[i][attr_wanted].keys() :
                multi = 0 
            else:
                multi =  prob_table[i][attr_wanted][key]
            prob_of_outcome = prob_of_outcome * multi

        prob_of_outcome = prob_of_outcome /(result_y**(len(tested)-2)*(total) ) 
        predicted_answer[key] = prob_of_outcome
    return max(predicted_answer, key= predicted_answer.get )

# Create outcome matrix: [[True positive, False positive],[False Negative, True Negative]]
def create_roc_matrix(prob_table, tested, parameter,total):

    matrix_keys ={}
    starting = 0
    for key in sorted(prob_table[parameter].keys()):
        matrix_keys[key] = starting
        starting += 1

    roc_matrix = [[0]*len(matrix_keys) for _ in range(len(matrix_keys))]
    for subject in tested:
        predicted = make_prediction(prob_table, subject, parameter, total)
        real = subject[parameter]
        roc_matrix[matrix_keys[predicted]][matrix_keys[real]] += 1

    return roc_matrix

# Function does the cross validation for the data set
def cross_validation(data, div, parameter):
    random.shuffle(data)
    length = len(data)

    split_data = [data[x:x+int(length/div)] for x in range(0, len(data), int(length/div))]

    result = []
    for i in range(len(split_data)):

        tested = split_data[i]
        data_set = []
        for data in range(len(split_data)):
            if data != i :
                data_set += split_data[data]
        tab_prob = create_table_prob(data_set, parameter)

        result.append(create_roc_matrix(tab_prob,tested,parameter,len(data_set)))
    return result