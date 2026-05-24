# CDO Correlation Geometry — Interactive Demo

Local Streamlit app for visualizing default correlation in a 2-CDS pool.
Companion to: *A Geometric Interpretation of Default Correlation in CDO Tranches* (Jribi, 2019/2026).

## What it does

- Converts CDS spreads to default probabilities and Gaussian thresholds
- Plots the joint density contour with zone partition (4 zones for 2 borrowers)
- Computes zone probabilities and tranche expected losses (equity / senior)
- Supports three copula models:
  - **Gaussian** — exact zone probabilities via bivariate normal CDF
  - **Student-t** — Monte Carlo zone probabilities, symmetric tail dependence
  - **Clayton** — Monte Carlo zone probabilities, lower-tail dependence only (defaults cluster in stress)

## Quick start

```bash
cd 002-cdo-correlation-geometry/demo
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Controls

| Parameter | Range | Default |
|-----------|-------|---------|
| Input Mode | CDS Spreads / Direct Probabilities | CDS Spreads |
| CDS Spread 1 | 10–2000 bps | 100 bps |
| CDS Spread 2 | 10–2000 bps | 100 bps |
| Recovery Rate | 0–80% | 40% |
| Horizon | 1–10 years | 5 years |
| Copula Model | Gaussian / Student-t / Clayton | Gaussian |
| Correlation (rho) | 0.00–0.99 | 0.50 (Gaussian, Student-t) |
| Degrees of Freedom (nu) | 2–30 | 5 (Student-t only) |
| Dependence (theta) | 0.1–20.0 | 2.0 (Clayton only) |

## What to look for

1. **Equal spreads, rho = 0**: all four zones have equal probability (25% each at p = 50%)
2. **Increase rho toward 0.99**: zones 2 and 3 (single defaults) drain; zones 1 and 4 (none/both) dominate
3. **Switch to Student-t**: contours become fatter-tailed (symmetric), zone 4 probability increases relative to Gaussian at the same rho
4. **Switch to Clayton**: contours become asymmetric (pear-shaped toward lower-left). Higher theta pushes more mass into zone 4 (both default) without affecting the upper tail — this is lower-tail dependence
5. **Total expected loss**: always equals p1 + p2 regardless of copula model — dependence redistributes loss, it does not create or destroy it

## Dependencies

- Python 3.9+
- streamlit, numpy, scipy, matplotlib
