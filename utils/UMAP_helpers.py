from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import umap

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42


def _fit_umap(X, n_neighbors=15, min_dist=0.1, metric="cosine", seed=100, verbose=True):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=verbose,
    )
    print(f"Fitting UMAP on {X.shape[0]} points ...")
    coords = reducer.fit_transform(X)
    print(f"Done. Shape: {coords.shape}")
    return coords


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


def umap_plot_from_embeddings(
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
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    output_dir="plots",
    output_filename=None,
    seed=100,
    verbose=True,
    show=True,
):
    X = emb_data if isinstance(emb_data, np.ndarray) else emb_data[embedding_key]["embeddings"]
    coords = _fit_umap(X, n_neighbors=n_neighbors, min_dist=min_dist,
                       metric=metric, seed=seed, verbose=verbose)

    if color_by == "gender":
        fig = _plot_gender(coords, df, n_sample, seed)
        default_fname = "umap_gender.pdf"
    elif color_by == "occupation":
        fig = _plot_occupation(coords, df, n_sample, top_n_occupations, seed)
        default_fname = "umap_occupation.pdf"
    elif color_by == "birth_year":
        fig = _plot_birth_year(
            coords, df, n_sample, seed,
            year_min=year_min, year_pivot=year_pivot,
            year_max=year_max, pivot_cmap_frac=pivot_cmap_frac,
            xlabel="UMAP-1", ylabel="UMAP-2",
            title="UMAP: coloured by birth year (nonlinear scale)",
        )
        default_fname = "umap_birth_year.pdf"
    elif color_by == "gender_x_occupation":
        fig = _plot_gender_x_occupation(coords, df, n_sample, top_n_occupations, seed)
        default_fname = "umap_gender_x_occupation.pdf"

    elif color_by == "occupation_top5":
        fig = _plot_occupation_top5(coords, df, n_sample, top_n_occupations, seed)
        default_fname = "umap_occupation_top5.pdf"
    else:
        raise ValueError(f"color_by must be 'gender', 'occupation', or 'birth_year'; got {color_by!r}")

    if output_filename is None:
        fname = default_fname
    else:
        fname = output_filename if output_filename.endswith(".pdf") else output_filename + ".pdf"
    _save(fig, os.path.join(output_dir, fname))
    if show:
        plt.show()
    return fig


def _plot_gender(coords, df, n_sample, seed):
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
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP: coloured by gender")
    ax.legend(markerscale=3, fontsize=9)
    fig.tight_layout()
    return fig


def _plot_occupation(coords, df, n_sample, top_n, seed):
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
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP: coloured by occupation (top {top_n})")
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

def _plot_gender_x_occupation(coords, df, n_sample, top_n, seed):
    """Color points by (gender × occupation) cross, top_n occupations + 'other'."""
    first_occ = df["occupation"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown"
    )
    # Exclude "unknown" from top occupations
    top_occs = (
        first_occ[first_occ != "unknown"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    occ_label = first_occ.apply(lambda x: x if x in top_occs else "other").values
    gender_vals = df["gender"].values

    genders = ["male", "female"]

# Per-occupation color pairs: (male, female)
    occ_color_pairs = [
        ("#306e36", "#68e574"),  # occ 0: dark green / light green
        ("#810f02", "#c62614"),  # occ 1: dark red   / light red
        ("#23178E", "#5c4ceb"),  # occ 2: dark purple / light purple
    ]

    palette = {}
    for i, occ in enumerate(top_occs):
        palette[(occ, "male")]   = occ_color_pairs[i][0]
        palette[(occ, "female")] = occ_color_pairs[i][1]
    palette[("other", "male")]   = "#c0c0c0"
    palette[("other", "female")] = "#c0c0c0"
    palette[("other", "other")]  = "#c0c0c0"

    idx = _sample_idx(len(coords), n_sample, seed)
    occ_s    = occ_label[idx]
    gender_s = np.where(np.isin(gender_vals[idx], genders), gender_vals[idx], "other")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Background "other" points first
    mask_other = occ_s == "other"
    if mask_other.any():
        ax.scatter(
            coords[idx][mask_other, 0], coords[idx][mask_other, 1],
            c="#c0c0c0", s=4, alpha=0.15, label=None, rasterized=False,
        )

    # Foreground: each (occupation × gender) combination
    for occ in top_occs:
        for g in genders:
            mask = (occ_s == occ) & (gender_s == g)
            if mask.any():
                ax.scatter(
                    coords[idx][mask, 0], coords[idx][mask, 1],
                    c=palette[(occ, g)], s=7, alpha=0.65,
                    label=f"{occ} · {g}", rasterized=False,
                )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP: coloured by occupation × gender (top {top_n} occupations)")
    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def _plot_occupation_top5(coords, df, n_sample, top_n, seed):
    colors = ['#e57468', '#68e574', '#7468e5', '#e5d068', '#68d0e5']
    colors = ['#68e574', "#ff3300", "#ffd91a", '#68d0e5']


    first_occ = df["occupation"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "unknown"
    )
    # Exclude "unknown" before taking top_n
    valid_occ = first_occ[first_occ != "unknown"]
    top_occs  = valid_occ.value_counts().head(top_n).index.tolist()
    occ_label = first_occ.apply(lambda x: x if x in top_occs else "other").values

    palette = {occ: colors[i] for i, occ in enumerate(top_occs)}
    palette["other"]   = "#cccccc"
    palette["unknown"] = "#dddddd"

    idx = _sample_idx(len(coords), n_sample, seed)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Background: other + unknown first
    for bg in ("unknown", "other"):
        mask = occ_label[idx] == bg
        if mask.any():
            ax.scatter(coords[idx][mask, 0], coords[idx][mask, 1],
                       c=palette[bg], s=4, alpha=1, linewidths=0, edgecolors='black', rasterized=False)

    # Foreground: top occupations
    for occ in top_occs:
        mask = occ_label[idx] == occ
        if mask.any():
            ax.scatter(coords[idx][mask, 0], coords[idx][mask, 1],
                       c=palette[occ], s=12, alpha=1, label=occ,
                       linewidths=0.05, edgecolors='black', rasterized=False)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP: top {top_n} occupations")
    ax.legend(markerscale=3, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig