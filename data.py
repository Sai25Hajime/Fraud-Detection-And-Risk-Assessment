import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000
data = {
    'transaction_id': range(1, n_samples + 1),
    'amount': np.concatenate([np.random.normal(100, 50, 900), np.random.normal(1000, 500, 100)]),
    'time_of_day': np.random.randint(0, 24, n_samples),
    'is_fraud': np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
}
df = pd.DataFrame(data)
df.to_csv('transaction_data.csv', index=False)
