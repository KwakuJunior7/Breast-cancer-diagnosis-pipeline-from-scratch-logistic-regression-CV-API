# How it works

This is my walkthrough of the pipeline: what each step does, the maths behind it, and why I made each decision. File names point to where each piece lives in the code.

```
wdbc.data  ->  validate  ->  stratified 80/20 split
                                  |                \
                           training (455)        test (114)  <- locked away
                                  |
              repeated 5-fold CV (5 models, scaler inside each fold)
                                  |
            pick model (CV ROC-AUC)  ->  pick threshold (out-of-fold recall >= 0.98)
                                  |
                     refit on all 455 training rows
                                  |
                       evaluate ONCE on the 114 test rows
                                  |
                  save model.joblib + metadata.json  ->  API / Docker
```

## 1. Loading and validating the data (`src/wdbc/data.py`)

The raw file has no header. Column 1 is a patient ID, column 2 is the diagnosis (M or B), and columns 3–32 are 30 measurements: 10 properties of the cell nuclei (radius, texture, concavity and so on), each summarised as a mean, a standard error and a "worst" value.

I drop the ID. My original Octave script kept it inside `X`, but an ID is an arbitrary number. If IDs were assigned in batches by date or clinic, a model could learn the batch instead of the biology.

`validate()` checks the row count, duplicate IDs, labels, missing values and negative measurements, and stops with a clear error if anything is off. I'd rather the pipeline refuse to run than train on a corrupted download.

The split is stratified so that train and test both keep the 37% malignant rate. With only 42 malignant test cases, an unlucky random split could leave too few to measure recall properly.

## 2. Standardising features (`src/wdbc/preprocessing.py`)

This is the z-score from my Lecture 8 code, `z = (x − mean) / std`, with two changes:

1. **Fit on training rows only, then apply to everything.** If the mean and standard deviation included test rows, information from the test set would leak into training.
2. NumPy broadcasting replaces Octave's `ones(n,1)*x_mean`.

Scaling matters for gradient descent. Area runs into the thousands while smoothness is around 0.1, so without scaling the cost surface is a long narrow valley and gradient descent zig-zags slowly down it.

While porting the code I found a bug in the class version: it computed the correlation matrix as `Z'*Z/(N−1)` with N equal to the number of *columns*. The divisor should be the number of *rows* minus one; it only worked because the example matrix was 3 × 3. My version uses rows, and a test checks it against `numpy.corrcoef`.

## 3. Logistic regression from scratch (`src/wdbc/logistic_scratch.py`)

This is the gradient-descent loop from my linear-regression homework, adapted for classification:

| | Linear regression (coursework) | Logistic regression (this project) |
| --- | --- | --- |
| Prediction | y_p = X̂u | p = sigmoid(X̂u), a probability between 0 and 1 |
| Cost | C = 1/(2n) Σ e² | J = mean cross-entropy + (λ/2)‖w‖² |
| Gradient | −(1/n) X̂ᵀe | (1/n) X̂ᵀ(p − y) + λw |
| Update | u ← u − μ · grad | the same |
| Stopping | fixed 5 epochs | stop when \|J(k) − J(k−1)\| < 1e-10 |

**Why cross-entropy.** Squared error on a sigmoid output gives a non-convex cost with flat regions where the gradient vanishes. Cross-entropy is convex for logistic regression, so there's one global minimum. Its gradient also simplifies neatly. For one sample:

```
z = wᵀx + b,   p = σ(z),   σ'(z) = p(1 − p)
L = −[y log p + (1 − y) log(1 − p)]
dL/dp = −y/p + (1 − y)/(1 − p) = (p − y) / (p(1 − p))
dL/dz = dL/dp · dp/dz = p − y
dL/dw = (p − y) x,   dL/db = p − y
```

Averaging over the n samples and adding λw for the L2 penalty gives `_gradient()`.

**Numerical stability.** The naive `1/(1+exp(−z))` overflows for very negative z, so my sigmoid splits by the sign of z and never calls exp() on a large positive number. Taking log(p) when p rounds to exactly 0 gives −∞, so I compute the cost directly from z as `logaddexp(0, z) − y·z`. Both cases are tested.

**Choosing the step size.** Instead of guessing a learning rate, I compute the largest the gradient can change, L = λ_max(X̂ᵀX̂/n)/4 + λ, and use μ = 1/L. With that step, gradient descent is guaranteed to lower the cost at every iteration. Here L ≈ 3.3, so μ ≈ 0.30, and training converges in 1,611 iterations. One test confirms the cost never rises; another shows that a step of 50 makes it oscillate.

**Regularisation (λ = 0.01).** Fifteen pairs of features are almost duplicates (|r| ≥ 0.95). Radius, perimeter and area are essentially the same measurement. With near-duplicates, many different weight combinations fit equally well and the weights can drift large. The penalty (λ/2)‖w‖² picks the smallest-weight solution and keeps the problem well-behaved. I don't penalise the bias, since shifting the decision boundary isn't overfitting.

**Proving it's correct.** `numerical_gradient()` estimates each partial derivative as (J(w+ε) − J(w−ε)) / 2ε, and a test checks that my formula agrees to five significant digits. Separately, scikit-learn minimises C·Σloss + ½‖w‖², which is the same problem as mine when C = 1/(λn). With that setting (`MatchedLogisticRegression`), the two models give identical cross-validation scores and matching coefficients.

## 4. The models I compared (`src/wdbc/models.py`)

Every candidate is a `Pipeline(scaler → classifier)`, so the scaler is refit inside each cross-validation fold.

- **Decision tree (depth 4).** Easy to read, but unstable: its CV ROC-AUC varies about ten times more across folds than the others.
- **Random forest.** Averages 300 trees to reduce that instability.
- **SVM with RBF kernel.** A flexible curved boundary. SVMs output distances, not probabilities, so I wrap it in `CalibratedClassifierCV` to convert them.
- **Two logistic regressions:** mine, and scikit-learn's on the same objective.

## 5. Cross-validation (`src/wdbc/evaluate.py`)

I use stratified 5-fold cross-validation repeated 3 times, for 15 train/validate splits. A single 80/20 split, as in my course final, gives one noisy number; 15 splits give a mean and a spread.

Every training row also gets an **out-of-fold** prediction from a model that never saw it, averaged over the 3 repeats. Those honest predictions are what I use to choose the threshold, so the test set stays untouched.

## 6. Picking the model (`src/wdbc/train.py`)

I rank models by mean cross-validation ROC-AUC, which doesn't depend on any threshold. Models within 0.002 of the best count as tied, because the fold-to-fold spread is about 0.005 and smaller gaps are noise. Among tied models I take the simplest. My logistic regression ties scikit-learn's exactly and is listed first, so it's deployed.

## 7. Picking the threshold (`choose_threshold`)

The two mistakes have different costs:

- **False negative:** a cancer is called benign, and diagnosis is delayed.
- **False positive:** a benign tumour is flagged, and the patient has a follow-up test.

So I set a recall target of 0.98 and take the highest threshold whose out-of-fold recall meets it. The highest such threshold gives the fewest false alarms while still meeting the target. The result is 0.27.

On the test set this changes 2 missed cancers into 1, at the cost of 3 false alarms instead of 0. The one cancer still missed scores 0.12. Catching it would mean flagging 7 of 72 benign patients instead of 3, and that cut-off was never validated, so I don't tune towards it.

## 8. The single test evaluation

The test set is scored once, after every decision is made. Looking at test results and then changing the model would make the test score optimistic.

One honest note: while building this, an earlier version used λ = 0.001, which converged too slowly (it hit the 20,000-iteration cap). I raised λ to 0.01 to fix convergence. The cross-validation ROC-AUC was essentially unchanged (0.9956 vs 0.9955), so the change is justified by CV and convergence, but I had already seen test numbers by then. Tuning λ inside nested cross-validation would remove that concern, and it's first on my list of next steps.

## 9. Serving predictions (`src/wdbc/predict.py`, `src/wdbc/api.py`)

`Predictor` loads the saved model together with `metadata.json`, which holds the threshold, the feature list and the training range of each feature. It rejects missing, extra, negative, NaN and infinite inputs. Values outside the training range are scored but flagged, because a model is least reliable where it has seen no data.

The FastAPI request schema is generated from the same feature list used in training, so the API can't drift out of sync with the model. `GET /health` lets Docker and cloud platforms check that the service is alive, and `GET /` redirects to the interactive docs.

## 10. Engineering around the model

- **`configs/config.yaml`** holds every number, so experiments change config, not code.
- **Tests** cover data validation, the scaler maths, the gradient check, equivalence with scikit-learn, convergence, threshold logic, end-to-end train → predict, and the API.
- **GitHub Actions** runs lint, tests and a full training run on every push, then builds the Docker image and smoke-tests it.
- **Docker** trains during the build, runs as a non-root user, and has a health check.
- **Run tracking** uses MLflow when it's installed and falls back to JSON logs in `reports/runs/`.

## Reading the charts

| Figure | What it shows |
| --- | --- |
| `class_balance.png` | 357 benign, 212 malignant. Always guessing "benign" would score 63% accuracy, which is why I don't rely on accuracy. |
| `feature_separation.png` | The six features that best separate the classes. Size and concavity of the worst nuclei lead (Cohen's d ≈ 2.6). |
| `correlation_heatmap.png` | R = ZᵀZ/(n − 1). Radius, perimeter and area are near-duplicates (r ≈ 0.99), which is why I regularise. |
| `cv_roc_pr.png` | Out-of-fold ROC and precision-recall curves. The linear models match the best; the single tree is clearly worse. |
| `threshold_tradeoff.png` | Precision and recall as the threshold moves. Lowering it catches more cancers and raises false alarms; I stop where recall reaches 0.98. |
| `test_confusion.png` | 41 of 42 malignant caught, 3 false alarms. |
| `gd_cost_history.png` | Cost against iteration on a log scale: a steady decrease to convergence in about 1,600 steps. |

**About the weights.** The largest weights on the standardised features are texture_worst (+0.81), radius_se (+0.66), radius_worst (+0.62) and symmetry_worst (+0.62); positive weights push towards malignant. Because so many features are correlated, I don't read individual weights as importance scores. Permutation importance or SHAP would be the right tool for that.
