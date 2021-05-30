import pandas as pd

train_csv = pd.read_csv('train.csv')
print(train_csv.head(10))
train_data = []
list_dataset = train_csv.values.tolist()
for dataset in list_dataset:
    text = dataset[1]
    labels = dataset[2]
    train_data.append([text, labels])
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
print(train_df.head(10))