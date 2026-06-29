import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

N   = 8  
DIM = 2048  
SEED = 42

rng = np.random.default_rng(SEED)

photos = rng.standard_normal((N, DIM)).astype(np.float32)
photos /= np.linalg.norm(photos, axis=1, keepdims=True)

queries = []
for i in range(N - 2):
    queries.append([i + 1, i + 2, i + 3])         
queries.append([1, 4, 7])                            
queries.append([2, 5, 8])                          
queries.append([1, 3, 6, 8])                        

with open(DATA_DIR / "photos.csv", "w") as f:
    for vec in photos:
        f.write(",".join(f"{v:.8f}" for v in vec) + "\n")

with open(DATA_DIR / "queries.csv", "w") as f:
    for q in queries:
        f.write(",".join(map(str, q)) + "\n")

print(f"Dataset sintetico creato:")
print(f"  {N} foto  ({DIM} dimensioni)")
print(f"  {len(queries)} query")
print(f"  File: data/photos.csv, data/queries.csv")
print()
print("Ora puoi lanciare:")
print("  python experiments/run.py")
print()
print("Per tornare al dataset reale, rimetti i file originali in data/")
