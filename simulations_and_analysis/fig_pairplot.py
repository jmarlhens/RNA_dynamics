import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# ── 1 · Fake posterior draws (5 000 samples) ─────────────────────────────────
rng = np.random.default_rng(42)
mu = np.array([90.0, 16.1, 7.8e-4])  # ktx, ktl, kRNA_deg
sigma = np.diag((0.10 * mu) ** 2)  # 10 % SD, no correlations
posterior_df = pd.DataFrame(
    rng.multivariate_normal(mu, sigma, 5_000), columns=["ktx", "ktl", "kRNA_deg"]
)
cols = posterior_df.columns.tolist()

# ── 2 · Style: bigger fonts, plain white bg ──────────────────────────────────
facet = 1.6  # inches per facet
sns.set_theme(
    style="white",
    font="DejaVu Sans",
    font_scale=0.95,  # ↑ overall fonts
    rc={"axes.labelweight": "bold", "axes.labelsize": 10},
)

# ── 3 · Pairplot with thick contour lines, no fill ───────────────────────────
g = sns.pairplot(
    posterior_df,
    kind="kde",
    diag_kind="kde",
    height=facet,
    aspect=1.0,
    plot_kws=dict(
        fill=False,  # <-- no shaded fill
        color="black",
        bw_adjust=0.8,
        levels=6,  # 6 contour levels
        linewidths=1.4,  # thicker lines
    ),
    diag_kws=dict(
        color="black",
        fill=True,  # keep diagonal filled for contrast
        alpha=0.25,
        bw_adjust=0.8,
        linewidth=0,
    ),
)

# ── 4 · Hide tick numbers, add only outer labels ─────────────────────────────
for ax in g.axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

for j, col in enumerate(cols):  # bottom-row x-labels
    g.axes[-1, j].set_xlabel(col, fontsize=10, labelpad=4)

for i, col in enumerate(cols):  # left-col y-labels
    g.axes[i, 0].set_ylabel(col, fontsize=10, labelpad=8)
    g.axes[i, 0].yaxis.set_label_coords(-0.20, 0.5)

# ── 5 · Square canvas & tidy title spacing ───────────────────────────────────
# side = facet * len(cols) + 0.9                    # space for labels/title
# g.fig.set_size_inches(side, side)
#
# g.fig.suptitle("Posterior KDE Pairplot of Kinetic Rates",
#                fontsize=11, y=0.965, weight="bold")
#
# g.fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10, top=0.90)

# ── 6 · Save PNG ─────────────────────────────────────────────────────────────
g.fig.savefig("kinetic_rates_posterior_pairplot.png", dpi=300, bbox_inches="tight")
plt.close(g.fig)

print("Saved figure as kinetic_rates_posterior_pairplot.png")
