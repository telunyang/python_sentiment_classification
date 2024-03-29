import pandas as pd
import matplotlib.pyplot as plt

path = './your-model-path/training_progress_scores.csv'
df = pd.read_csv(path)

df = df[ df['global_step'] % 2000 != 0]
df.insert(0, 'epoch', range(1, 1 + len(df)))
# df['epoch'] = df.index + 1

print(df[ ['epoch', 'precision', 'recall', 'f1_score'] ])

plot = df.plot.line(x='epoch', y=['train_loss', 'eval_loss'])
plt.title('Sentiment Classification')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'eval_loss'], loc='upper right')
plt.ylim(0.0, 1.0) 
plt.show()