from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42


def _fit_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=100)
    coords = pca.fit_transform(X)
    return coords, pca


def _sample_idx(n_total, n_sample, seed=100):
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=min(n_sample, n_total), replace=False)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")


def _make_birth_year_norm(year_min, year_pivot, year_max, pivot_cmap_frac):
    class PiecewiseNorm(mpl.colors.Normalize):
        def __call__(self, value, clip=None):
            x = np.asarray(value, dtype=float)
            out = np.empty_like(x)
            lo = x < year_pivot
            hi = ~lo
            out[lo] = np.interp(x[lo], [year_min, year_pivot], [0.0, pivot_cmap_frac])
            out[hi] = np.interp(x[hi], [year_pivot, year_max], [pivot_cmap_frac, 1.0])
            return np.ma.array(out, mask=np.isnan(x))

    return PiecewiseNorm(vmin=year_min, vmax=year_max)


def pca_plot_from_embeddings(
    emb_data,
    df,
    *,
    color_by="gender",
    embedding_key="sentence_template_trivial",
    n_sample=15_000,
    top_n_occupations=10,
    year_min=800,
    year_pivot=1600,
    year_max=2000,
    pivot_cmap_frac=0.15,
    output_dir="plots",
    output_filename=None,
    seed=100,
    show=True,
):
    X = emb_data if isinstance(emb_data, np.ndarray) else emb_data[embedding_key]["embeddings"]
    coords, pca = _fit_pca(X)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    if color_by == "gender":
        fig = _plot_gender(coords, df, var1, var2, n_sample, seed)
    elif color_by == "occupation":
        fig = _plot_occupation(coords, df, var1, var2, n_sample, top_n_occupations, seed)
    elif color_by == "birth_year":
        fig = _plot_birth_year(
            coords, df, n_sample, seed,
            year_min=year_min, year_pivot=year_pivot,
            year_max=year_max, pivot_cmap_frac=pivot_cmap_frac,
            xlabel=f"PC1 ({var1:.1f}%)", ylabel=f"PC2 ({var2:.1f}%)",
            title="PCA: coloured by birth year (nonlinear scale)",
        )
    else:
        raise ValueError(f"color_by must be 'gender', 'occupation', or 'birth_year'; got {color_by!r}")

    if output_filename is None:
        fname = f"pca_{color_by}.pdf"
    else:
        fname = output_filename if output_filename.endswith(".pdf") else output_filename + ".pdf"
    _save(fig, os.path.join(output_dir, fname))
    if show:
        plt.show()
    return fig


def scree_plot_from_embeddings(
    emb_data,
    *,
    embedding_key="sentence_template_trivial",
    n_components=30,
    n_components_cumulative=100,
    output_dir="plots",
    show=True,
):
    X = emb_data if isinstance(emb_data, np.ndarray) else emb_data[embedding_key]["embeddings"]
    n_fit = min(max(n_components, n_components_cumulative), X.shape[0], X.shape[1])
    n_components = min(n_components, n_fit)
    n_components_cumulative = min(n_components_cumulative, n_fit)
    _, pca = _fit_pca(X, n_components=n_fit)

    evr  = pca.explained_variance_ratio_ * 100
    cumr = np.cumsum(evr)

    bar_color = "#4C72B0"
    line_color = "darkorange"

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    pcs_scree = np.arange(1, n_components + 1)
    ax1.bar(pcs_scree, evr[:n_components], color=bar_color, alpha=0.75)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title(f"Scree plot — {embedding_key}")
    fig1.tight_layout()
    _save(fig1, os.path.join(output_dir, "scree_plot.pdf"))

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    pcs_cum = np.arange(1, n_components_cumulative + 1)
    ax2.plot(pcs_cum, cumr[:n_components_cumulative], color=line_color, marker="o", ms=3, lw=1.8)
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_title(f"Cumulative variance — {embedding_key}")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(0, n_components_cumulative + 1)
    ax2.grid(True, linestyle="--", alpha=0.5)
    fig2.tight_layout()
    _save(fig2, os.path.join(output_dir, "scree_plot_cumulative.pdf"))

    if show:
        plt.show()
    return fig1, fig2


def _plot_gender(coords, df, var1, var2, n_sample, seed):
    idx         = _sample_idx(len(coords), n_sample, seed)
    gender_vals = df["gender"].values[idx]
    categories  = ["male", "female"]
    palette     = {"male": "steelblue", "female": "darkorange", "other": "#cccccc"}

    fig, ax = plt.subplots(figsize=(8, 6))
    mask_other = ~np.isin(gender_vals, categories)
    ax.scatter(coords[idx][mask_other, 0], coords[idx][mask_other, 1],
               c=palette["other"], s=5, alpha=0.25, label="other", rasterized=False)
    for g in categories:
        mask = gender_vals == g
        ax.scatter(coords[idx][mask, 0], coords[idx][mask, 1],
                   c=palette[g], s=5, alpha=0.45, label=g, rasterized=False)
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title("PCA: coloured by gender")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_occupation(coords, df, var1, var2, n_sample, top_n, seed):
    first_occ = df["occupation"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown"
    )
    top_occs  = first_occ.value_counts().head(top_n).index.tolist()
    occ_label = first_occ.apply(lambda x: x if x in top_occs else "other").values

    cmap_occ = plt.cm.get_cmap("tab20", top_n)
    palette  = {occ: cmap_occ(i) for i, occ in enumerate(top_occs)}
    palette["other"]   = "#cccccc"
    palette["unknown"] = "#eeeeee"

    idx = _sample_idx(len(coords), n_sample, seed)

    fig, ax = plt.subplots(figsize=(10, 7))
    for bg in ("unknown", "other"):
        mask = occ_label[idx] == bg
        ax.scatter(coords[idx][mask, 0], coords[idx][mask, 1],
                   c=palette[bg], s=4, alpha=0.2, label=bg, rasterized=False)
    for occ in top_occs:
        mask = occ_label[idx] == occ
        ax.scatter(coords[idx][mask, 0], coords[idx][mask, 1],
                   color=palette[occ], s=5, alpha=0.55, label=occ, rasterized=False)
    ax.set_xlabel(f"PC1 ({var1:.1f}%)")
    ax.set_ylabel(f"PC2 ({var2:.1f}%)")
    ax.set_title(f"PCA: coloured by occupation (top {top_n})")
    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def _plot_birth_year(
    coords, df, n_sample, seed,
    year_min=800, year_pivot=1600, year_max=2000, pivot_cmap_frac=0.15,
    xlabel="Dim-1", ylabel="Dim-2", title="Coloured by birth year (nonlinear scale)",
):
    rng        = np.random.default_rng(seed)
    birth_year = pd.to_numeric(df["birth_year"], errors="coerce").values[:len(coords)]
    has_year   = np.isfinite(birth_year)
    idx        = rng.choice(np.where(has_year)[0], size=min(n_sample, has_year.sum()), replace=False)

    norm = _make_birth_year_norm(year_min, year_pivot, year_max, pivot_cmap_frac)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[idx, 0], coords[idx, 1],
                    c=birth_year[idx], cmap="viridis", norm=norm,
                    s=5, alpha=0.5, rasterized=False)
    cb = plt.colorbar(sc, ax=ax, label="Birth year")
    cb.ax.axhline(year_pivot, color="white", linewidth=1.2, linestyle="--")
    cb.ax.text(1.35, year_pivot, f"← {year_pivot}", va="center", fontsize=7,
               transform=cb.ax.get_yaxis_transform(), color="dimgrey")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig
