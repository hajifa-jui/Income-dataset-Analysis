
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/adult.csv')



df.describe()
# print(df.info())

df.head()

sns.countplot(df['Income'])

df['Gender'].value_counts().plot(kind='pie',autopct='%.2f')

import matplotlib.pyplot as plt
plt.hist(df['Age'],bins=5)

sns.distplot(df['Age'])

sns.boxplot(df['Age'])

df['Age'].min()

df['Age'].max()

df['Age'].mean()

df['Age'].skew()



sns.scatterplot(Capital Gain['CapitalGain'],CapitalGain['CapitalGain'],hue=df['Gender'],style=df['Race'],size=df['size'])

sns.barplot(['Education'],['Age'],hue=['CapitalGain'])
