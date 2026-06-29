# Data Reduction Practical

Implementation of four photo-deletion strategies guided by a query log.

## Methods

- **Method A** (`brute_force.py`): Exhaustive search with cosine similarity. Optimal but intractable for large N.
- **Method B** (`indepdf.py`): IndepDF greedy algorithm with Jaccard similarity.
- **Method C** (`shapley.py`): Selection by Shapley value. Exact for N ≤ 15, closed-form approximation otherwise.
- **Method D** (`ewc.py`): Entropy-Weighted Coverage (EWC).

## Setup

```bash
pip install -r requirements.txt
```

Place `photos.csv` and `queries.csv` in `data/`.

## Usage

```bash
# Full run, skip Method A (intractable for large N)
python experiments/run.py --skip-a --budget-fraction 0.05

# Run all methods on small synthetic dataset
python experiments/run.py
```

Results and plots are saved in `experiments/output/`.
