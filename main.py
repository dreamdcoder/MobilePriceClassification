import pandas as pd
data = pd.read_csv("Dataset\\train.csv")

from sdv.tabular import GaussianCopula
model = GaussianCopula()
model.fit(data)
sample = model.sample(1000)
sample.to_csv("Dataset\\test2.csv")
print(sample.head())


conda activate C:\Anaconda\envs\mltech
C:\Anaconda\Scripts\conda.exe install -p C:/Anaconda/envs/mltech plyer -y

