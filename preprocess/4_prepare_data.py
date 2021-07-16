
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

df = pd.read_csv('data/pre-processed/audio_features.csv')
df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
print(df.shape)
display(df.head())

# change 7 to 2
df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
df.head()

# df.to_csv('data/pre-processed/no_sample_df.csv')

# oversample fear
fear_df = df[df['label']==3]
for i in range(30):
    df = df.append(fear_df)

sur_df = df[df['label']==4]
for i in range(10):
    df = df.append(sur_df)
    
# df.to_csv('data/pre-processed/modified_df.csv')


scalar = MinMaxScaler()
df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])
df.head()

x_train, x_test = train_test_split(df, test_size=0.20)

x_train.to_csv('data/s2e/audio_train.csv', index=False)
x_test.to_csv('data/s2e/audio_test.csv', index=False)

print(x_train.shape, x_test.shape)