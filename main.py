import os
import pickle
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "swxg")

import swxg
from WeatherGenerator import WeatherGenerator
from compare_generated_and_empirical_distributions import (
    compare_weather,
    validate_with_plot,
    validation_metrics_table,
)

# =====================================================
# MAIN
# =====================================================

def main():
    lat = 12.873030556839481
    lon = 53.9152712861946

    date_start = "1990-01-01"
    date_end = "2025-12-31"

    model = WeatherGenerator(
        method="stochastic",
        lon=lon,
        lat=lat,
        date_start=date_start,
        date_end=date_end,
        cache="disk",
    )

    model.fit(seed=42)

    historical = model.retrieve_weather(
        lon,
        lat,
        date_start,
        date_end,
        cache="disk",
        method="open_meteo",
    )

    years = 10

    for _ in range(5):
        df = model.generate(years)

    print("finished")

    compare_weather(
        historical,
        df,
        n_days=365 * 10,
    )

    # =================================================
    # DISTRIBUTION VALIDATION
    # =================================================

    validation_cols = (
        "TEMP_MIN",
        "TEMP_MAX",
        "PRECIP",
        "IRRADIANCE",
        "ETP",
    )

    results = validate_with_plot(
        hist_df=historical,
        synth_df=df,
        cols=validation_cols,
        n_bins=50,
        precip_log=True,
    )

    validation_metrics_table(
        hist_df=historical,
        synth_df=df,
        cols=validation_cols,
    )

    return df


if __name__ == "__main__":
    df = main()