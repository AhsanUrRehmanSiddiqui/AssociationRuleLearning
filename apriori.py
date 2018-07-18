import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#import tkinker
import matplotlib.pyplot as plt
import pandas as pd
from apyori import  apriori

dataset = pd.read_csv("GroceryStoreDataSet.csv")
transactions = []
row, col = dataset.shape
print("row", row)
print("col", col)

for i in range(row):
    transactions.append([str(dataset.values[i,j])for j in range(col)])


#training apriori on the dateset

rules = apriori(transactions, min_support=0.00000001, min_confidence=0.0000001, min_lift=1, min_length=1)

#visualizing the results
results = list(rules)

for i in range(len(results)):
    print(results[i])
    print('-'*20)
#print(results)

