import matplotlib.pyplot as plt
import random
import numpy

from bayes import get_data, cross_validation

# Calculate the True Positive Rate
def calculate_tpr(roc_matrix):
    return roc_matrix[0][0]/(roc_matrix[0][0] +roc_matrix[1][0] )

# Calculate the False Positive Rate
def calculate_fpr(roc_matrix):
    return roc_matrix[0][1]/(roc_matrix[1][1] + roc_matrix[0][1] )

# Calculate the Accuracy
def calculate_acc(roc_matrix):
    return ((roc_matrix[0][0]+ roc_matrix[1][1])/(roc_matrix[1][1] + roc_matrix[0][1]+roc_matrix[1][0] + roc_matrix[0][0] ))

# Draw graph for given results from one or many data sets
def make_roc(title,rocs_stats):
    results = make_plot_read(rocs_stats)
    plt.subplots(1, figsize=(10,10))
    plt.title(title)
    x = results[0]
    y = results[1]
    plt.plot(x , y) #FPR then TPR

    plt.plot([0, 1], ls="--")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(title + ".png")
    plt.show()

# Sums the results of all tests
def sum_results(results):
    final_sum = [[0]*len(results[0]) for _ in range(len(results[0]))]
    for item in results:
        for i in range(len(results[0])):
            for j in range(len(results[0])):
                final_sum[i][j] += item[i][j]
    return final_sum

# Mae the results (TPR and FPR) readable for the plot
def make_plot_read(data):
    data_read = [[0],[0]]
    for outcome in data:
        data_read[0].append(outcome[0])
        data_read[1].append(outcome[1])
    
    data_read[0].append(1)
    data_read[1].append(1)

    return data_read

# Draws results for given data set or data sets
# Prints the Accuracy for the given data set
# data_sets - [file_name, parameter, times repeated]

def get_results(title, *data_sets):
    results_matr = []
    for sets in data_sets:
        pre_results = []
        for _ in range(sets[2]):
            got_data = get_data(sets[0])
            result = cross_validation(got_data, 3, sets[1])
            pre_results.append(sum_results(result))

        final_result = sum_results(pre_results)
        print("Accuracy:",sets[0], "=",round(calculate_acc(final_result),2))
        results_matr.append([calculate_fpr(final_result), calculate_tpr(final_result)])

    results_matr.sort(key = lambda x: x[0])
    make_roc(title,results_matr)
