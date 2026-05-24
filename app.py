"""
2-CDS Correlation Geometry Visualizer

Interactive Streamlit app demonstrating the geometric interpretation
of default correlation from:
  F. Jribi, "A Geometric Interpretation of Default Correlation in CDO Tranches"

Three copula models:
  - Gaussian copula (elliptical contours, no tail dependence)
  - Student-t copula (symmetric tail dependence)
  - Clayton copula (lower-tail dependence only — defaults cluster in stress)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from scipy.stats import norm, multivariate_normal, t as student_t


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def cds_to_threshold(spread_bps, recovery, horizon):
    """Convert CDS spread (bps) to Gaussian default threshold."""
    s = spread_bps / 10_000  # bps to decimal
    lam = s / (1 - recovery)
    p = 1 - np.exp(-lam * horizon)
    p = np.clip(p, 1e-8, 1 - 1e-8)
    D = norm.ppf(p)
    return p, D


def gaussian_zone_probs(D1, D2, rho):
    """Exact zone probabilities under Gaussian copula."""
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    rv = multivariate_normal(mean, cov)
    # P(Z4) = P(X1 < D1, X2 < D2)
    P_Z4 = rv.cdf([D1, D2])
    # P(X1 < D1) = p1, P(X2 < D2) = p2
    p1 = norm.cdf(D1)
    p2 = norm.cdf(D2)
    P_Z2 = p1 - P_Z4
    P_Z3 = p2 - P_Z4
    P_Z1 = 1 - P_Z2 - P_Z3 - P_Z4
    return P_Z1, P_Z2, P_Z3, P_Z4


def gaussian_joint_pdf(x, y, rho):
    """Bivariate Gaussian PDF."""
    det = 1 - rho**2
    exponent = -(x**2 + y**2 - 2 * rho * x * y) / (2 * det)
    return np.exp(exponent) / (2 * np.pi * np.sqrt(det))


def studentt_joint_pdf(x, y, rho, nu):
    """Bivariate Student-t PDF with correlation rho and nu d.f."""
    det = 1 - rho**2
    Q = (x**2 + y**2 - 2 * rho * x * y) / det
    coeff = 1 / (2 * np.pi * np.sqrt(det))
    return coeff * (1 + Q / nu) ** (-(nu + 2) / 2) * \
        np.exp(
            np.log(np.math.gamma((nu + 2) / 2))
            - np.log(np.math.gamma(nu / 2))
            - np.log(nu * np.pi)
        ) * (2 * np.pi)  # normalize: Gamma ratio * (nu*pi)^{-1} * det^{-1/2}


def studentt_joint_pdf_proper(x, y, rho, nu):
    """Bivariate Student-t PDF (correct normalization)."""
    from scipy.special import gammaln
    det = 1 - rho**2
    Q = (x**2 + y**2 - 2 * rho * x * y) / det
    log_coeff = (
        gammaln((nu + 2) / 2)
        - gammaln(nu / 2)
        - np.log(nu * np.pi)
        - 0.5 * np.log(det)
    )
    log_pdf = log_coeff - ((nu + 2) / 2) * np.log(1 + Q / nu)
    return np.exp(log_pdf)


def studentt_zone_probs(D1, D2, rho, nu, n_samples=50_000):
    """Zone probabilities under Student-t copula via Monte Carlo.

    Student-t copula: generate bivariate t samples, then check
    which zone each falls in (using Gaussian marginal thresholds
    after applying the probability integral transform).
    """
    rng = np.random.default_rng(42)
    # Generate bivariate t: X = Z / sqrt(W/nu) where Z ~ N(0, Sigma), W ~ chi2(nu)
    cov = np.array([[1, rho], [rho, 1]])
    Z = rng.multivariate_normal([0, 0], cov, size=n_samples)
    W = rng.chisquare(nu, size=n_samples)
    T_samples = Z / np.sqrt(W[:, None] / nu)  # bivariate t

    # Student-t copula: transform marginals to uniform via t-CDF, then to Gaussian
    U1 = student_t.cdf(T_samples[:, 0], df=nu)
    U2 = student_t.cdf(T_samples[:, 1], df=nu)
    X1 = norm.ppf(np.clip(U1, 1e-10, 1 - 1e-10))
    X2 = norm.ppf(np.clip(U2, 1e-10, 1 - 1e-10))

    # Zone classification
    z4 = np.sum((X1 < D1) & (X2 < D2))
    z2 = np.sum((X1 < D1) & (X2 >= D2))
    z3 = np.sum((X1 >= D1) & (X2 < D2))
    z1 = np.sum((X1 >= D1) & (X2 >= D2))

    total = z1 + z2 + z3 + z4
    return z1 / total, z2 / total, z3 / total, z4 / total


def studentt_copula_pdf(x, y, rho, nu):
    """Student-t copula density in Gaussian-marginal space.

    For plotting: evaluate the copula density at points (x, y) that are
    in the Gaussian latent space. We need to transform to uniform, then
    to t, evaluate the t density, and divide by the product of Gaussian
    marginal densities.
    """
    from scipy.special import gammaln

    # Transform Gaussian -> uniform -> t
    u1 = norm.cdf(x)
    u2 = norm.cdf(y)
    u1 = np.clip(u1, 1e-10, 1 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1 - 1e-10)
    t1 = student_t.ppf(u1, df=nu)
    t2 = student_t.ppf(u2, df=nu)

    # Bivariate t density
    det = 1 - rho**2
    Q = (t1**2 + t2**2 - 2 * rho * t1 * t2) / det
    log_f2 = (
        gammaln((nu + 2) / 2)
        - gammaln(nu / 2)
        - np.log(nu * np.pi)
        - 0.5 * np.log(det)
        - ((nu + 2) / 2) * np.log(1 + Q / nu)
    )
    # Marginal t densities
    log_f1_t1 = student_t.logpdf(t1, df=nu)
    log_f1_t2 = student_t.logpdf(t2, df=nu)

    # Copula density = f2 / (f1 * f1) * gaussian_marginals
    # But we want the joint density in Gaussian-marginal space:
    # f(x,y) = c(F(x), F(y)) * phi(x) * phi(y)
    # where c is the copula density
    log_copula = log_f2 - log_f1_t1 - log_f1_t2
    log_joint = log_copula + norm.logpdf(x) + norm.logpdf(y)

    return np.exp(log_joint)


def clayton_zone_probs(D1, D2, theta, n_samples=50_000):
    """Zone probabilities under Clayton copula via Monte Carlo.

    Clayton copula: C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}.
    Lower-tail dependence only — defaults cluster in stress.
    """
    rng = np.random.default_rng(42)
    # Marshall–Olkin algorithm for Clayton copula sampling
    # 1. V ~ Gamma(1/theta, 1)
    # 2. E1, E2 ~ Exp(1) independent
    # 3. U_i = (1 + E_i / V)^{-1/theta}
    V = rng.gamma(1.0 / theta, 1.0, size=n_samples)
    E1 = rng.exponential(1.0, size=n_samples)
    E2 = rng.exponential(1.0, size=n_samples)
    U1 = (1 + E1 / V) ** (-1.0 / theta)
    U2 = (1 + E2 / V) ** (-1.0 / theta)

    # Transform uniform marginals to Gaussian
    U1 = np.clip(U1, 1e-10, 1 - 1e-10)
    U2 = np.clip(U2, 1e-10, 1 - 1e-10)
    X1 = norm.ppf(U1)
    X2 = norm.ppf(U2)

    # Zone classification
    z4 = np.sum((X1 < D1) & (X2 < D2))
    z2 = np.sum((X1 < D1) & (X2 >= D2))
    z3 = np.sum((X1 >= D1) & (X2 < D2))
    z1 = np.sum((X1 >= D1) & (X2 >= D2))

    total = z1 + z2 + z3 + z4
    return z1 / total, z2 / total, z3 / total, z4 / total


def clayton_copula_pdf(x, y, theta):
    """Clayton copula density in Gaussian-marginal space.

    For plotting: the joint density is c(Phi(x), Phi(y)) * phi(x) * phi(y),
    where c is the Clayton copula density.

    c(u,v) = (1+theta) * (u*v)^{-(1+theta)} * (u^{-theta} + v^{-theta} - 1)^{-(2+1/theta)}
    """
    u = norm.cdf(x)
    v = norm.cdf(y)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)

    # Clayton copula density in log space for stability
    log_c = (
        np.log(1 + theta)
        - (1 + theta) * (np.log(u) + np.log(v))
        - (2 + 1.0 / theta) * np.log(u**(-theta) + v**(-theta) - 1)
    )

    # Joint density in Gaussian-marginal space
    log_joint = log_c + norm.logpdf(x) + norm.logpdf(y)

    # Handle numerical issues where u^{-theta} + v^{-theta} - 1 <= 0
    valid = (u**(-theta) + v**(-theta) - 1) > 0
    result = np.zeros_like(x)
    result[valid] = np.exp(log_joint[valid])
    return result


def limiting_zone_probs(p1, p2):
    """Zone probabilities at rho -> 1 (perfect correlation)."""
    if p1 <= p2:
        return 1 - p2, 0.0, p2 - p1, p1
    else:
        return 1 - p1, p1 - p2, 0.0, p2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_contour_plot(D1, D2, rho, model, nu=5, theta=2.0, zone_probs=None):
    """Generate the 2D contour plot with zone partition."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=120)

    # Grid
    grid_range = 3.5
    x = np.linspace(-grid_range, grid_range, 250)
    y = np.linspace(-grid_range, grid_range, 250)
    X, Y = np.meshgrid(x, y)

    # Joint density
    if model == "Gaussian":
        Z = gaussian_joint_pdf(X, Y, rho)
    elif model == "Student-t":
        Z = studentt_copula_pdf(X, Y, rho, nu)
    else:  # Clayton
        Z = clayton_copula_pdf(X, Y, theta)

    # Contour plot — levels span from near-zero to just above the peak
    levels = np.linspace(Z.max() * 0.01, Z.max() * 1.05, 15)
    cmap = plt.cm.YlOrRd
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.85, extend='max')
    ax.contour(X, Y, Z, levels=levels[:-1], colors='darkred', linewidths=0.3, alpha=0.5)

    # Threshold lines
    ax.axvline(D1, color='#2166ac', linewidth=1.5, linestyle='--',
               label=f'$D_1 = {D1:.2f}$')
    ax.axhline(D2, color='#1b7837', linewidth=1.5, linestyle='--',
               label=f'$D_2 = {D2:.2f}$')

    # Zone labels with probabilities
    label_style = dict(fontsize=7, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor='gray', alpha=0.85))
    if zone_probs is not None:
        pZ1, pZ2, pZ3, pZ4 = zone_probs
        z1_lbl = f'$Z_1$\nno default\n{pZ1:.1%}'
        z2_lbl = f'$Z_2$\nonly 1\n{pZ2:.1%}'
        z3_lbl = f'$Z_3$\nonly 2\n{pZ3:.1%}'
        z4_lbl = f'$Z_4$\nboth\n{pZ4:.1%}'
    else:
        z1_lbl, z2_lbl, z3_lbl, z4_lbl = '$Z_1$', '$Z_2$', '$Z_3$', '$Z_4$'
    # Z1: top-right (no default)
    ax.text((D1 + grid_range) / 2, (D2 + grid_range) / 2, z1_lbl, **label_style)
    # Z2: top-left (only 1 defaults)
    ax.text((D1 - grid_range) / 2, (D2 + grid_range) / 2, z2_lbl, **label_style)
    # Z3: bottom-right (only 2 defaults)
    ax.text((D1 + grid_range) / 2, (D2 - grid_range) / 2, z3_lbl, **label_style)
    # Z4: bottom-left (both default)
    ax.text((D1 - grid_range) / 2, (D2 - grid_range) / 2, z4_lbl, **label_style)

    ax.set_xlabel('$X_1$ (Obligor 1)', fontsize=7)
    ax.set_ylabel('$X_2$ (Obligor 2)', fontsize=7)
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=6)
    ax.legend(loc='upper right', fontsize=6)

    if model == "Gaussian":
        model_label = "Gaussian"
        param_str = f"rho={rho:.2f}"
    elif model == "Student-t":
        model_label = f"Student-t (nu={nu})"
        param_str = f"rho={rho:.2f}"
    else:
        model_label = "Clayton"
        param_str = f"theta={theta:.1f}"
    ax.set_title(f'{model_label}, {param_str}',
                 fontsize=8, fontweight='bold')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CDO Correlation Geometry",
    page_icon="",
    layout="wide"
)

# Reduce top padding
st.markdown(
    "<style>div.block-container {padding-top: 2rem;}</style>",
    unsafe_allow_html=True,
)

st.subheader("2-CDS Correlation Geometry Visualizer")
st.caption(
    "Interactive demonstration of the geometric interpretation of default "
    "correlation. Adjust CDS spreads, recovery, and copula parameters to see "
    "how the joint density, zone probabilities, and tranche expected losses change."
)

# Sidebar
st.sidebar.header("Input Mode")
input_mode = st.sidebar.radio(
    "Choose input mode",
    ["CDS Spreads", "Direct Probabilities"],
    help="CDS Spreads: derive default probabilities from market inputs. "
         "Direct Probabilities: set p₁, p₂ directly (e.g. to reproduce paper examples)."
)

if input_mode == "CDS Spreads":
    st.sidebar.header("Market Inputs")
    s1 = st.sidebar.slider("CDS Spread 1 (bps)", 10, 2000, 100, step=5)
    s2 = st.sidebar.slider("CDS Spread 2 (bps)", 10, 2000, 100, step=5)
    R = st.sidebar.slider("Recovery Rate", 0.0, 0.80, 0.40, step=0.05)
    T = st.sidebar.slider("Horizon (years)", 1, 10, 5)
    p1, D1 = cds_to_threshold(s1, R, T)
    p2, D2 = cds_to_threshold(s2, R, T)
else:
    st.sidebar.header("Default Probabilities")
    p1 = st.sidebar.slider("p₁ (default prob obligor 1)", 0.01, 0.99, 0.50, step=0.01)
    p2 = st.sidebar.slider("p₂ (default prob obligor 2)", 0.01, 0.99, 0.50, step=0.01)
    D1 = norm.ppf(p1)
    D2 = norm.ppf(p2)

st.sidebar.header("Copula Model")
model = st.sidebar.selectbox(
    "Model", ["Gaussian", "Student-t", "Clayton"],
    help="Gaussian: elliptical, no tail dependence. "
         "Student-t: symmetric tail dependence. "
         "Clayton: lower-tail dependence only (defaults cluster in stress)."
)
nu = 5
theta = 2.0
if model in ("Gaussian", "Student-t"):
    rho = st.sidebar.slider("Correlation (rho)", 0.0, 0.99, 0.50, step=0.01)
    if model == "Student-t":
        nu = st.sidebar.slider("Degrees of freedom (nu)", 2, 30, 5)
else:
    rho = 0.0  # not used for Clayton, but needed for reference computations
    theta = st.sidebar.slider(
        "Dependence (theta)", 0.1, 20.0, 2.0, step=0.1,
        help="Higher theta = stronger lower-tail dependence. "
             "theta -> 0: independence. theta -> inf: comonotonicity."
    )

# Display derived quantities
st.sidebar.markdown("---")
st.sidebar.markdown("**Derived Quantities**")
st.sidebar.markdown(f"$p_1$ = {p1:.4f}  |  $D_1$ = {D1:.3f}")
st.sidebar.markdown(f"$p_2$ = {p2:.4f}  |  $D_2$ = {D2:.3f}")

# Compute zone probabilities
if model == "Gaussian":
    P_Z1, P_Z2, P_Z3, P_Z4 = gaussian_zone_probs(D1, D2, rho)
elif model == "Student-t":
    P_Z1, P_Z2, P_Z3, P_Z4 = studentt_zone_probs(D1, D2, rho, nu)
else:  # Clayton
    P_Z1, P_Z2, P_Z3, P_Z4 = clayton_zone_probs(D1, D2, theta)

# Limiting probabilities
P_Z1_0, P_Z2_0, P_Z3_0, P_Z4_0 = gaussian_zone_probs(D1, D2, 0.0)
P_Z1_lim, P_Z2_lim, P_Z3_lim, P_Z4_lim = limiting_zone_probs(p1, p2)

# Tranche expected losses
equity_EL = P_Z2 + P_Z3 + P_Z4
senior_EL = P_Z4
total_EL = p1 + p2

equity_EL_0 = P_Z2_0 + P_Z3_0 + P_Z4_0
senior_EL_0 = P_Z4_0
equity_EL_lim = P_Z2_lim + P_Z3_lim + P_Z4_lim
senior_EL_lim = P_Z4_lim

# Layout
col_plot, col_tables = st.columns([3, 2])

with col_plot:
    fig = make_contour_plot(D1, D2, rho, model, nu, theta,
                            zone_probs=(P_Z1, P_Z2, P_Z3, P_Z4))
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with col_tables:
    st.subheader("Zone Probabilities")
    st.markdown(
        "| Zone | Current | At rho=0 | At rho->1 |\n"
        "|:-----|--------:|---------:|----------:|\n"
        f"| $Z_1$ (no default) | {P_Z1:.4f} | {P_Z1_0:.4f} | {P_Z1_lim:.4f} |\n"
        f"| $Z_2$ (only 1) | {P_Z2:.4f} | {P_Z2_0:.4f} | {P_Z2_lim:.4f} |\n"
        f"| $Z_3$ (only 2) | {P_Z3:.4f} | {P_Z3_0:.4f} | {P_Z3_lim:.4f} |\n"
        f"| $Z_4$ (both) | {P_Z4:.4f} | {P_Z4_0:.4f} | {P_Z4_lim:.4f} |\n"
        f"| **Total** | **{P_Z1+P_Z2+P_Z3+P_Z4:.4f}** | **1.0000** | **1.0000** |"
    )

    st.subheader("Tranche Expected Losses")
    st.caption("Unit notionals, zero recovery. Equity = [0, 1], Senior = [1, 2].")
    direction_eq = "long rho" if equity_EL < equity_EL_0 else "short rho"
    direction_sr = "short rho" if senior_EL > senior_EL_0 else "long rho"
    st.markdown(
        "| Tranche | Current | At rho=0 | At rho->1 | Direction |\n"
        "|:--------|--------:|---------:|----------:|:----------|\n"
        f"| **Equity** (1st default) | {equity_EL:.4f} | {equity_EL_0:.4f} "
        f"| {equity_EL_lim:.4f} | {direction_eq} |\n"
        f"| **Senior** (2nd default) | {senior_EL:.4f} | {senior_EL_0:.4f} "
        f"| {senior_EL_lim:.4f} | {direction_sr} |\n"
        f"| **Total** | {equity_EL + senior_EL:.4f} | {total_EL:.4f} "
        f"| {total_EL:.4f} | invariant |"
    )

    st.subheader("Geometric Interpretation")
    if model == "Gaussian":
        contour_desc = "ellipse (symmetric)"
    elif model == "Student-t":
        contour_desc = "fat-tailed contour (symmetric tail dependence)"
    else:
        contour_desc = "pear-shaped contour (lower-tail dependence only)"
    st.markdown(
        f"- **Spread risk**: translates threshold lines ($D_1$, $D_2$)\n"
        f"- **Correlation risk**: deforms the density contour — {contour_desc}\n"
        f"- **Zero-sum**: total E[L] = $p_1 + p_2$ = {total_EL:.4f} "
        f"regardless of dependence structure"
    )

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------

st.markdown("---")

with st.expander("How it works"):
    st.markdown(
        "Each obligor's creditworthiness is modelled as a latent variable "
        "$X_i$. Default occurs when $X_i$ falls below a threshold $D_i$ "
        "derived from the obligor's CDS spread (or set directly).\n\n"
        "The **copula** controls the joint distribution of $(X_1, X_2)$:\n"
        "- **Gaussian**: elliptical contours, no tail dependence.\n"
        "- **Student-t**: heavier tails, symmetric tail dependence "
        "(both upper and lower).\n"
        "- **Clayton**: asymmetric, lower-tail dependence only "
        "-- defaults cluster in stress, but survivals do not.\n\n"
        "The two threshold lines $D_1$, $D_2$ partition the plane into "
        "four **zones**:\n"
        "- $Z_1$: neither defaults. $Z_2$: only obligor 1 defaults. "
        "$Z_3$: only obligor 2 defaults. $Z_4$: both default.\n\n"
        "**Key insight**: changing the copula (or its parameters) "
        "redistributes probability mass across zones, but the total "
        "expected loss $p_1 + p_2$ is invariant. "
        "This is the *zero-sum principle* of correlation trading: "
        "the equity tranche is long correlation, the senior tranche "
        "is short correlation."
    )

# ---------------------------------------------------------------------------
# Links (filled in after publishing)
# ---------------------------------------------------------------------------

PAPER_URL = "https://www.researchgate.net/publication/405210525_A_Geometric_Interpretation_of_Default_Correlation_in_CDO_Tranches"
MEDIUM_URL = None  # Medium article link (set after publishing)

links = []
if PAPER_URL:
    links.append(f"[Paper]({PAPER_URL})")
if MEDIUM_URL:
    links.append(f"[Medium article]({MEDIUM_URL})")
links.append(
    "[Source code](https://github.com/fjribi/cdo-correlation-demo)"
)

st.markdown("---")
st.markdown(
    "*Based on: F. Jribi, \"A Geometric Interpretation of Default Correlation "
    "in CDO Tranches\" (2019/2026).*"
)
if links:
    st.markdown(" | ".join(links))

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

st.caption(
    "This is an educational visualization tool, not a pricer. "
    "It may contain errors or simplifications. "
    "Feedback and issues: "
    "[GitHub](https://github.com/fjribi/cdo-correlation-demo/issues)"
)

