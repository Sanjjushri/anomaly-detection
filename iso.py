import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/mnt/data/trade_dataset_medium.csv")

# Select only relevant numeric features for anomaly detection
features = [
    'price', 'volume', 'volatility', 'rsi', 'macd', 'bollinger_position',
    'market_cap', 'pe_ratio', 'volume_profile', 'price_momentum',
    'liquidity_ratio', 'order_book_imbalance', 'cross_correlation',
    'sector_beta', 'vwap_deviation', 'time_of_day_factor'
]

X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['anomaly_score'] = iso_forest.decision_function(X_scaled)
df['predicted_anomaly'] = iso_forest.predict(X_scaled)

# Convert prediction: -1 is anomaly, 1 is normal
df['predicted_anomaly'] = df['predicted_anomaly'].map({1: 0, -1: 1})

# View some predicted anomalies
anomalies = df[df['predicted_anomaly'] == 1]

# Display counts and a heatmap for visualization
print("Anomaly Count:", df['predicted_anomaly'].value_counts())

# Heatmap of anomaly scores vs actual
sns.histplot(df['anomaly_score'], kde=True)
plt.title("Anomaly Score Distribution")
plt.show()
