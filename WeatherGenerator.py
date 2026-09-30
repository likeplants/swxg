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
warnings.filterwarnings("ignore")


# =====================================================
# SSP SCENARIOS
# =====================================================

SSP_DEFAULTS = {
    "SSP1-2.6": {"temp_delta": 1.5, "precip_scale": 1.05},
    "SSP2-4.5": {"temp_delta": 2.5, "precip_scale": 1.10},
    "SSP5-8.5": {"temp_delta": 4.0, "precip_scale": 1.20},
}

SUPPORTED_SYNTH_METHODS = {
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP5-8.5",
    "stochastic",
}

SUPPORTED_FORMATS = {
    "default",
    "pystics",
}

# =====================================================
# WEATHER GENERATOR
# =====================================================

class WeatherGenerator:

    def __init__(
        self,
        method: str,
        weather: dict = None,
        lon: float = None,
        lat: float = None,
        date_start: str = None,
        date_end: str = None,
        cache: str | None = "memory",
    ):
        self.method = method
        self.cache = cache

        dataframe_mode = weather is not None

        if dataframe_mode:
            if any([lon, lat, date_start, date_end]):
                raise Exception(
                    "Provide either weather OR API parameters, not both."
                )
        else:
            if lon is None or lat is None:
                raise Exception(
                    "You must provide both lon and lat for API mode."
                )

            if not date_start or not date_end:
                raise Exception(
                    "You must provide date_start and date_end for API mode."
                )

        self.lon = lon
        self.lat = lat
        self.date_start = date_start
        self.date_end = date_end
        self.dataframe_mode = dataframe_mode

        self._model = None

        if dataframe_mode:
            self.weather_df = weather
        else:
            self.weather_df = self.retrieve_weather(
                lon,
                lat,
                date_start,
                date_end,
                cache,
                method="open_meteo",
            )

    # =================================================
    # WEATHER RETRIEVAL
    # =================================================

    def retrieve_weather(
        self,
        lon,
        lat,
        date_start,
        date_end,
        cache,
        method,
    ):
        if method != "open_meteo":
            raise NotImplementedError(
                "Only open_meteo is supported atm."
            )

        def _historical_weather_key(
            lon,
            lat,
            date_start,
            date_end,
        ):
            return lon, lat, date_start, date_end

        def _historical_weather_filename(key):
            lon, lat, date_start, date_end = key

            return (
                f"historical_weather_"
                f"{lon}_{lat}_{date_start}_{date_end}.csv"
            )

        key = _historical_weather_key(
            lon,
            lat,
            date_start,
            date_end,
        )

        # Memory cache
        if cache == "memory":
            if not hasattr(self, "_historical_weather_cache"):
                self._historical_weather_cache = {}

            if key in self._historical_weather_cache:
                return self._historical_weather_cache[key]

        # Disk cache
        cache_dir = Path.cwd() / "cache"

        if cache == "disk":
            cache_dir.mkdir(exist_ok=True)

            cache_file = cache_dir / _historical_weather_filename(key)

            if cache_file.exists():
                df = pd.read_csv(
                    cache_file,
                    parse_dates=["DATETIME"],
                )

                print(
                    "Historical weather successfully loaded from disk: "
                    + str(cache_file)
                )

                return df

        # Fetch data
        df = self.fetch_open_meteo(
            [lat, lon],
            site=str([lat, lon]),
            start_date=date_start,
            end_date=date_end,
        )

        print("successfully fetched weather from open meteo")

        # Save cache
        if cache == "memory":
            self._historical_weather_cache[key] = df

        elif cache == "disk":
            cache_dir.mkdir(exist_ok=True)
            df.to_csv(cache_file, index=False)

        return df

    # =================================================
    # OPEN METEO
    # =================================================

    def fetch_open_meteo(
        self,
        lat_lon,
        site="SITE",
        start_date=None,
        end_date=None,
    ):
        lat, lon = lat_lon

        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Europe/Berlin",
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration",
            ]),
        }

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        daily = response.json()["daily"]

        df = pd.DataFrame({
            "SITE": site,
            "DATETIME": pd.to_datetime(
                daily["time"],
                errors="raise",
            ).astype("datetime64[ns]"),
            "PRECIP": daily.get("precipitation_sum"),
            "TEMP": daily.get(
                "temperature_2m_mean",
                [
                    (mx + mn) / 2
                    for mx, mn in zip(
                        daily["temperature_2m_max"],
                        daily["temperature_2m_min"],
                    )
                ],
            ),
            "TEMP_MIN": daily["temperature_2m_min"],
            "TEMP_MAX": daily["temperature_2m_max"],
            "ETP": daily["et0_fao_evapotranspiration"],
            "IRRADIANCE": daily.get("shortwave_radiation_sum"),
        })

        return df

    # =================================================
    # CACHE
    # =================================================

    def clear_cache(self, cache_type="all"):
        """
        Clear model/weather caches.

        Parameters
        ----------
        cache_type : str
            "memory" | "disk" | "all"
        """

        cache_dir = Path.cwd() / "cache"

        # Memory cache
        if cache_type in ("memory", "all"):

            if hasattr(self, "_model_cache"):
                self._model_cache.clear()

            if hasattr(self, "_historical_weather_cache"):
                self._historical_weather_cache.clear()

        # Disk cache
        if cache_type in ("disk", "all"):

            if cache_dir.exists():
                for file in cache_dir.glob("*"):
                    if file.is_file():
                        file.unlink()

    # =================================================
    # MODEL
    # =================================================

    def fit(self, seed):
        self.model = self._get_model(seed)

    def _get_model(self, seed):

        def _model_cache_key():
            return (
                self.lon,
                self.lat,
                self.date_start,
                self.date_end,
                seed,
                self.dataframe_mode,
            )

        def _model_cache_filename(key):
            (
                lon,
                lat,
                date_start,
                date_end,
                seed,
                dataframe_mode,
            ) = key

            return (
                f"swxg_model_"
                f"{lon}_{lat}_{date_start}_{date_end}_"
                f"{seed}_{dataframe_mode}.pkl"
            )

        key = _model_cache_key()

        # Cache disabled
        if self.cache is None:
            random.seed(seed)
            np.random.seed(seed)

            model = swxg.SWXGModel(self.weather_df)
            model.fit()

            return model

        # Memory cache
        if self.cache == "memory":

            if not hasattr(self, "_model_cache"):
                self._model_cache = {}

            if key in self._model_cache:
                return self._model_cache[key]

        # Disk cache
        elif self.cache == "disk":

            cache_dir = Path.cwd() / "cache"
            cache_dir.mkdir(exist_ok=True)

            cache_file = cache_dir / _model_cache_filename(key)

            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    model = pickle.load(f)

                print(
                    "Model successfully loaded from disk: "
                    + str(cache_file)
                )

                return model

        # Build model
        random.seed(seed)
        np.random.seed(seed)

        model = swxg.SWXGModel(self.weather_df)
        model.fit()

        # Save cache
        if self.cache == "memory":
            self._model_cache[key] = model

        elif self.cache == "disk":
            with open(cache_file, "wb") as f:
                pickle.dump(model, f)

        return model

    # =================================================
    # GENERATION
    # =================================================

    def generate(
        self,
        years: int = 10,
        seed=42,
        method="stochastic",
    ):
        if method not in SUPPORTED_SYNTH_METHODS:
            raise Exception(
                "Method not available. Available methods: "
                + str(SUPPORTED_SYNTH_METHODS)
            )

        if method == "stochastic":
            df = self._stochastic_generate(years)
        else:
            df = self._ssp_generate(years, seed)

        return df

    def _ssp_generate(self, years, seed):
        model = self.model

        if not model:
            raise Exception("Call fit first")

        future = model.synthesize(
            n=years,
            validate=False,
        )

        scenario = SSP_DEFAULTS[self.method]

        future = future.copy()

        future["TEMP_MIN"] += scenario["temp_delta"]
        future["TEMP_MAX"] += scenario["temp_delta"]

        future["PRECIP"] *= scenario["precip_scale"]

        rng = np.random.default_rng(seed)

        num_cols = future.select_dtypes(
            include=[np.number]
        ).columns

        future[num_cols] += rng.normal(
            0,
            0.25,
            future[num_cols].shape,
        )

        return future.reset_index(drop=True)

    def _stochastic_generate(self, years):
        model = self.model

        if not model:
            raise Exception("Call fit first")

        future = model.synthesize(
            n=years,
            validate=False,
        )

        return future