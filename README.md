## Project Overview
When smartphone storage runs low, deleting the oldest or largest files ignores search history. This project models photo deletion as a subset selection problem. We evaluate four strategies to decide which photos to delete while preserving the ones most relevant to future searches.

### Implemented Methods
- **Method A (Exhaustive Search):** Computes the exact global optimum. Feasible only for small synthetic datasets.
- **Method B (IndepDF):** A greedy algorithm based on Jaccard similarity that scores photos by retrieval frequency.
- **Method C (Shapley Value):** Selects photos by maximizing the sum of their individual Shapley values.
- **Method D (Entropy-Weighted Coverage - EWC):** A novel strategy proposed in this project that scores photos by combining query frequency with the informational diversity (Shannon entropy) of the covered queries.

## Repository Structure
- `algorithms/`: Implementations of the four methods (`brute_force.py`, `indepdf.py`, `shapley.py`, `ewc.py`).
- `core/`: Core logic for data parsing, similarity computation, and dataset handling (`data.py`).
- `experiments/`: Main execution script to benchmark the algorithms (`run.py`).
- `data/`: Directory for the `.csv` datasets (ignored by version control due to size constraints).
- `create_synthetic.py`: Generates the $N=8$ synthetic dataset used for exact validation.
- `make_plots.py`: Generates the performance and utility plots (e.g., score distributions, utility trade-offs).

## Setup and Execution
1. **Install dependencies:**
   Ensure you have Python 3.12+ installed, then run:
   ```bash
   pip install -r requirements.txt
