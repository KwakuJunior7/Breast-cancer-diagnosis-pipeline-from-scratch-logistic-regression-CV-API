# Where this project started

These are the Octave scripts from my CSCI 5740 *Machine Learning Algorithms* class (Fall 2025, New Mexico Highlands University, taught by Dr. Gil Gallegos). I keep them here to show where the project began.

| Script | What it did | Where the idea lives now |
| --- | --- | --- |
| `CSCS4740_5740_MLA_EDA_wdbc.m` | Loaded `wdbc.data` and separated features from labels | `src/wdbc/data.py` (ID column dropped, validation added) and `src/wdbc/eda.py` |
| `cs_5740_mla_l8_z_score_9182025.m` | Computed z-scores and the correlation matrix R = Z'Z/(N−1) | `src/wdbc/preprocessing.py` (separate fit and transform; divisor corrected to rows − 1) |

The gradient-descent update rule comes from the linear-regression work in Lectures 18–24 and the final exam. [HOW_IT_WORKS.md](../HOW_IT_WORKS.md), section 3, compares the two side by side.
