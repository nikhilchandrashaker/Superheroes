# 🦸 Superhero Analytics: What Actually Makes a Hero?

A statistically rigorous exploration of a synthetic superhero dataset — testing (and mostly rejecting) the intuitive assumptions about what drives power, alignment, and public perception.

## Project Summary

This project takes a fun/synthetic Kaggle-style dataset (1,200 superheroes/villains, 17 features) and treats it with the same rigor as a real-world analysis: every hypothesis is tested formally rather than assumed, and null results are reported as clearly as positive ones. The standout finding is that most "obvious" relationships in the data (training → power, abilities → power, casualties → approval) **do not hold** — with one clear exception: **heroes train significantly more than villains**.

That's the actual point of the project: demonstrating the discipline to test assumptions, report honest negative results, and still surface the one real signal — rather than cherry-picking whatever looks interesting.

## Dataset

`superhero_dataset.csv` — 1,200 rows, 17 columns:

| Category | Columns |
|---|---|
| Physical | `height_cm`, `weight_kg`, `age` |
| Career | `years_active`, `training_hours_per_week` |
| Behavior/outcome | `civilian_casualties_past_year`, `power_level`, `public_approval_rating` |
| Abilities (binary) | `super_strength`, `flight`, `energy_projection`, `telepathy`, `healing_factor`, `shape_shifting`, `invisibility`, `telekinesis` |
| Target | `is_good` (1 = hero, 0 = villain) |

No missing values, no duplicate rows. Ability columns each appear in ~29–32% of rows, consistent with independent random assignment rather than designed correlation with other features.

## Notebook Structure

`superhero_analytics.ipynb` is organized into five phases:

1. **Data Cleaning & Feature Engineering** — quality checks, outlier review, and engineered features: `total_abilities`, `bmi`, `training_hours_per_year`, `casualties_per_year_active`, and a composite `reputation_index` (weights disclosed as a design choice, not a fitted model).
2. **Exploratory Analysis** — class balance, feature distributions, ability prevalence, physical traits by alignment, and scatterplots for the two headline questions (training vs. power, casualties vs. approval).
3. **Statistical Testing** — 10 hypotheses tested with Pearson correlation, independent t-tests, one-way ANOVA, and chi-square tests, summarized in a results table.
4. **Machine Learning** — Logistic Regression, Decision Tree, Random Forest, and XGBoost trained to predict `is_good`, evaluated with accuracy, F1, ROC-AUC, and 5-fold cross-validation, benchmarked against the majority-class baseline.
5. **Explainability** — feature importance (Random Forest, XGBoost), logistic regression coefficients, and SHAP values to confirm which features the models actually rely on.

## Key Findings

| Question | Result |
|---|---|
| Training hours → power level | ❌ No effect (r ≈ 0.00) |
| Age / years active → power level | ❌ No effect |
| More abilities → higher power | ❌ No effect |
| Any individual ability → power level | ❌ No significant effect |
| Hero vs. villain: power level | ✅ Small but significant (villains slightly higher) |
| Hero vs. villain: public approval | ⚠️ Borderline (p ≈ 0.049) |
| Hero vs. villain: civilian casualties | ❌ No difference |
| Hero vs. villain: **training hours** | ✅ **Clear, significant effect** |
| Civilian casualties → public approval | ❌ No relationship |
| Any ability → alignment (`is_good`) | ❌ No association |

**ML result:** best model (Random Forest) reaches ROC-AUC ≈ 0.61 — meaningfully above the 0.50 random baseline, but far short of a strong classifier, which is exactly what the statistical testing would predict. Feature importance and SHAP both point to `training_hours_per_week` as the dominant driver, with everything else contributing little beyond noise.

## How to Run

**Requirements:**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap jupyter
```

**Run the notebook:**
```bash
jupyter notebook superhero_analytics.ipynb
```
or open it directly in JupyterLab / VS Code. All cells run top-to-bottom with no manual steps; figures are saved to the working directory as PNGs when executed.

**Files:**
- `superhero_analytics.ipynb` — full analysis notebook (already executed, outputs included)
- `superhero_dataset.csv` — source data
- `*.png` — figures generated during notebook execution (regenerated automatically on re-run)

## Why This Framing

A dataset like this invites a tempting but weak version of the project: assume the causal story ("training makes you stronger," "villains hurt more civilians and get punished for it") and cherry-pick charts that seem to support it. This project instead leads with the testing process, is transparent when a hypothesis fails, and reports the one relationship that actually replicates (training hours ↔ alignment) with appropriate statistical and model-based support. That's a more convincing signal of data science maturity than a notebook that only shows the results that "worked."
