import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =====================================================
# PLOTTING
# =====================================================
def compare_weather(
    historical_df,
    generated_df,
    temp_min_col="TEMP_MIN",
    temp_max_col="TEMP_MAX",
    precip_col="PRECIP",
    irradiance_col="IRRADIANCE",
    etp_col="ETP",
    n_days=None,
):
    historical_df = historical_df.copy()
    generated_df = generated_df.copy()

    if n_days is not None:
        historical_df = historical_df.iloc[:n_days]
        generated_df = generated_df.iloc[:n_days]

    historical_df["DAY"] = np.arange(len(historical_df))
    generated_df["DAY"] = np.arange(len(generated_df))

    fig, ax = plt.subplots(
        4,
        2,
        figsize=(16, 10),
        sharex="col",
    )

    datasets = [
        (historical_df, "Historical"),
        (generated_df, "Generated"),
    ]

    columns = [
        (temp_min_col, temp_max_col, "Temperature"),
        (precip_col, None, "Precipitation"),
        (irradiance_col, None, "Irradiance"),
        (etp_col, None, "ETP"),
    ]

    for row, (col1, col2, ylabel) in enumerate(columns):

        vals = [
            historical_df[col1],
            generated_df[col1],
        ]

        if col2:
            vals += [
                historical_df[col2],
                generated_df[col2],
            ]

        ymin = np.nanmin(vals)
        ymax = np.nanmax(vals)

        margin = 0.05 * (ymax - ymin) if ymax > ymin else 1

        for col, (df, title) in enumerate(datasets):

            ax[row, col].plot(
                df["DAY"],
                df[col1],
                label=col1,
            )

            if col2:
                ax[row, col].plot(
                    df["DAY"],
                    df[col2],
                    label=col2,
                )

                ax[row, col].legend(
                    loc="upper left",
                    fontsize=10,
                )

            ax[row, col].set_ylim(
                ymin - margin,
                ymax + margin,
            )

            ax[row, col].tick_params(
                labelsize=7,
            )

            if row == 0:
                ax[row, col].set_title(
                    title,
                    fontsize=10,
                )

            if col == 0:
                ax[row, col].set_ylabel(
                    ylabel,
                    fontsize=10,
                )

            if row < 3:
                ax[row, col].tick_params(
                    labelbottom=False,
                )

    xticks = np.arange(
        0,
        max(len(historical_df), len(generated_df)),
        365,
    )

    for col in range(2):
        ax[-1, col].set_xticks(xticks)
        ax[-1, col].set_xticklabels(
            [f"Year {i + 1}" for i in range(len(xticks))],
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        "time_series_empirical_and_generated.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def validate_with_plot(
    hist_df,
    synth_df,
    cols=(
        "TEMP_MIN",
        "TEMP_MAX",
        "PRECIP",
        "IRRADIANCE",
        "ETP",
    ),
    n_bins=50,
    precip_log=True,
    eps=1e-8,
):
    """
    Compare historical and generated empirical distributions.
    """

    results = {}

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=(12, 3 * len(cols)),
    )

    if len(cols) == 1:
        axes = [axes]

    for i, col in enumerate(cols):

        if col not in hist_df.columns:
            raise KeyError(
                f"Column '{col}' not found in historical data. "
                f"Available columns: {list(hist_df.columns)}"
            )

        if col not in synth_df.columns:
            raise KeyError(
                f"Column '{col}' not found in generated data. "
                f"Available columns: {list(synth_df.columns)}"
            )

        hist = hist_df[col].dropna().values
        synth = synth_df[col].dropna().values

        # =================================================
        # Statistical metrics
        # =================================================

        ks_stat, p_value = stats.ks_2samp(
            hist,
            synth,
        )

        wass = stats.wasserstein_distance(
            hist,
            synth,
        )

        hist_std = np.std(
            hist,
            ddof=1,
        )

        if hist_std > 0:
            normalized_w = wass / hist_std
        else:
            normalized_w = np.nan

        results[col] = {
            "KS_distance": float(ks_stat),
            "wasserstein": float(wass),
            "normalized_wasserstein": float(normalized_w),
            "p_value": float(p_value),
        }

        # =================================================
        # Histogram
        # =================================================

        ax = axes[i]

        all_vals = np.concatenate(
            [hist, synth]
        )

        if col == "PRECIP" and precip_log:

            pos = all_vals[all_vals > eps]

            if len(pos) > 10 and pos.max() > pos.min():

                bins_used = np.logspace(
                    np.log10(pos.min()),
                    np.log10(pos.max()),
                    n_bins + 1,
                )

                ax.set_xscale("log")
                xlabel = "PRECIP (log scale)"

            else:

                bins_used = n_bins
                xlabel = "PRECIP"

        else:

            bins_used = np.histogram_bin_edges(
                all_vals,
                bins=n_bins,
            )

            xlabel = col

        # =================================================
        # Histograms
        # =================================================

        ax.hist(
            hist,
            bins=bins_used,
            alpha=0.5,
            label="Historical",
            density=True,
        )

        ax.hist(
            synth,
            bins=bins_used,
            alpha=0.5,
            label="Generated",
            density=True,
        )

        # =================================================
        # Labels / typography
        # =================================================

        ax.set_xlabel(
            xlabel,
            fontsize=12,
             labelpad=8
        )

        ax.set_ylabel("PDF", fontsize=12)

        ax.text(0.99, 0.95, f"Normalized W = {normalized_w:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.text(0.99, 0.80, f"KS p = {p_value:.2e}",           transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.text(0.99, 0.65, f"KS D = {ks_stat:.3f}",           transform=ax.transAxes, ha="right", va="top", fontsize=10)
        
        ax.tick_params(
            axis="both",
            labelsize=8,
        )

        ax.legend(
            fontsize=10,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("pdfs_empirical_and_generated.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results


def validation_metrics_table(
    hist_df,
    synth_df,
    cols=(
        "TEMP_MIN",
        "TEMP_MAX",
        "PRECIP",
        "IRRADIANCE",
        "ETP",
    ),
):
    """
    Calculate validation metrics for all weather variables.

    Returns
    -------
    pandas.DataFrame
        Formatted validation metrics.
    """

    metrics = {
        "KS distance": {},
        "Wasserstein": {},
        "Normalized W": {},
        "KS p-value": {},
    }

    for col in cols:

        if col not in hist_df.columns:
            raise KeyError(
                f"Column '{col}' not found in historical data. "
                f"Available columns: {list(hist_df.columns)}"
            )

        if col not in synth_df.columns:
            raise KeyError(
                f"Column '{col}' not found in generated data. "
                f"Available columns: {list(synth_df.columns)}"
            )

        hist = hist_df[col].dropna().values
        synth = synth_df[col].dropna().values

        # =================================================
        # Metrics
        # =================================================

        ks_stat, p_value = stats.ks_2samp(
            hist,
            synth,
        )

        wass = stats.wasserstein_distance(
            hist,
            synth,
        )

        hist_std = np.std(
            hist,
            ddof=1,
        )

        if hist_std > 0:
            normalized_w = wass / hist_std
        else:
            normalized_w = np.nan

        # =================================================
        # Units
        # =================================================

        if col in ("TEMP_MIN", "TEMP_MAX"):
            unit = "°C"

        elif col == "PRECIP":
            unit = "mm"

        elif col == "IRRADIANCE":
            unit = "W/m²"

        elif col == "ETP":
            unit = "mm"

        else:
            unit = ""

        # =================================================
        # Store
        # =================================================

        metrics["KS distance"][col] = (
            f"{ks_stat:.4f} [-]"
        )

        metrics["Wasserstein"][col] = (
            f"{wass:.4f} {unit}".strip()
        )

        metrics["Normalized W"][col] = (
            f"{normalized_w:.4f} [-]"
        )

        metrics["KS p-value"][col] = (
            f"{p_value:.2e} [-]"
        )

    # =====================================================
    # Build table
    # =====================================================

    table = pd.DataFrame(metrics).T

    table.index.name = "Metric"

    table = table[list(cols)]

    print(table.to_string())

    return table