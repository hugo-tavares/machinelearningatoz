# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


# Setting a list of lists
transactions = []
for i in range (0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Traning Apriori on the dataset
from apyori import apriori

# min_support of a product that is purchased 3 times a day:
# 3 * 7 / 7500 = 0.0028
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the result
results = list(rules)

results_list = []

for i in range(0, len(results)):

    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\n' + str(results[i][2]))