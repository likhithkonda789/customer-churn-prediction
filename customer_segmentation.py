import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("customers.csv")

# Select features
X = df[['AnnualIncome', 'SpendingScore']]

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Display clustered data
print(df)

# Visualization
plt.figure()
plt.scatter(X['AnnualIncome'], X['SpendingScore'], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
