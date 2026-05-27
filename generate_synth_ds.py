import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 500

# Create empty arrays
X = np.random.normal(0, 1, n_samples)  # root
W = np.random.normal(0, 1, n_samples)  # root
Y = np.zeros(n_samples)
Z = np.zeros(n_samples)
V = np.zeros(n_samples)                # collider: Z -> V <- W

# Define the causal rules (Time Series / Lagged relationships)
for t in range(1, n_samples):
    # X causes Y with a lag of 1
    Y[t] = 0.8 * X[t-1] + np.random.normal(0, 0.5)

    # Y causes Z with a lag of 1
    Z[t] = 0.6 * Y[t-1] + np.random.normal(0, 0.5)

    # Z and W both cause V (collider)
    V[t] = 0.7 * Z[t-1] + 0.5 * W[t-1] + np.random.normal(0, 0.5)

# Save to CSV
df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'W': W, 'V': V})
df.to_csv('synthetic_causal_data.csv', index=False)
print("Saved synthetic_causal_data.csv!")