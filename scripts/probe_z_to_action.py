import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--z_path", type=str, required=True)
parser.add_argument("--actions_path", type=str, required=True)
parser.add_argument("--name", type=str, default="method")
args = parser.parse_args()

Z = np.load(args.z_path)
A = np.load(args.actions_path)

n = min(len(Z), len(A))
Z = Z[:n]
A = A[:n]

Z_train, Z_test, A_train, A_test = train_test_split(
    Z, A, test_size=0.25, random_state=0
)

model = Ridge(alpha=1.0)
model.fit(Z_train, A_train)
A_pred = model.predict(Z_test)

mse = mean_squared_error(A_test, A_pred)
r2 = r2_score(A_test, A_pred)

# Direction cosine similarity
num = (A_test * A_pred).sum(axis=1)
den = (np.linalg.norm(A_test, axis=1) * np.linalg.norm(A_pred, axis=1)) + 1e-12
cos = num / den

print(f"== {args.name} ==")
print("Z shape:", Z.shape)
print("Actions shape:", A.shape)
print("Probe MSE:", mse)
print("Probe R2:", r2)
print("Mean action cosine:", cos.mean())
