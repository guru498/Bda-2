import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

# Load dataset
data = pd.read_csv('weatherHistory.csv')

# Drop unnecessary columns and handle missing values
data = data.drop(['Formatted Date', 'Summary', 'Precip Type', 'Daily Summary'], axis=1)
data = data.dropna()

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# SOM initialization
som = MiniSom(x=10, y=10, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Train the SOM
som.random_weights_init(data_scaled)
som.train_random(data_scaled, num_iteration=1000)

# Plotting the SOM
plt.figure(figsize=(10, 8))
plt.pcolor(som.distance_map().T, cmap='coolwarm')
plt.colorbar()
plt.title('Self-Organizing Map - Distance Map')

# Correct relative path for saving output
plt.savefig('output/som_outputs.png')
plt.show()
