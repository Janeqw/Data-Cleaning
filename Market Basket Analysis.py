import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder


data = pd.read_csv('C:/Users/siwan/OneDrive/桌面/Xia Resume/data analysis project/Market Basket Analysis/DataSets/online_retail.csv')
items = data['Description'].astype(str)
stocks = items.apply(lambda t : t.split(',')).tolist()

# Instantiate transaction encoder and identify unique items in transactions
encoder= TransactionEncoder().fit(stocks)
onehot = encoder.transform(stocks)

# Convert one-hot encoded data to DataFrame
onehot = pd.DataFrame(onehot,columns=encoder.columns_)

categorise data




#choose frequent items:
frequent_sets = apriori(onehot, min_support=0.005, use_colnames=True, max_len=100)
print(frequent_sets)




