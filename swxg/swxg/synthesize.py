import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as stats
import statsmodels
import warnings

from .make_figures import *

def synthesize_data(n: int,
                    data: pd.DataFrame,
                    precip_dict: dict,
                    copula_dicts: dict,
                    resolution: str,
                    validate: bool,
                    dirpath: str,
                    synthesize_kwargs: dict) -> pd.DataFrame:
    # collecting kNN years
    global kNN_dict
    kNN_dict = {"precip": [], "temp": {}}

    # synthesize kwargs    
    default_synthesize_kwargs = {"validation_samplesize_mult": 10,
                                 "figure_extension": "svg"}
    if not synthesize_kwargs: 
        synthesize_kwargs = default_synthesize_kwargs
    else:
        for k in default_synthesize_kwargs:
            if k not in synthesize_kwargs:
                synthesize_kwargs[k] = default_synthesize_kwargs[k]
    
    # validation
    global do_validation, validation_dirpath, validation_extension
    do_validation, validation_dirpath, validation_extension = validate, dirpath, synthesize_kwargs["figure_extension"]
     
    # remove all the rows with partially full or unfit years
    filtered_data = data[data["YEAR"].isin(precip_dict["log10_annual_precip"].index.values)]
    incomplete_years, full_years = [], []
    for year in sorted(set(filtered_data["YEAR"].values)):
        if len(set(filtered_data.loc[filtered_data["YEAR"] == year, "MONTH"].values)) == 12:
            full_years.append(int(year))
        else:
            incomplete_years.append(int(year))
    filtered_data = filtered_data[filtered_data["YEAR"].isin(full_years)]
    filtered_data.reset_index(drop=True, inplace=True)
    if resolution == "daily" and "DAY" in filtered_data.columns:
        filtered_data.astype({"SITE": str, "YEAR": int, "MONTH": int, "DAY": int, "PRECIP": float, "TEMP": float})
    else:
        filtered_data.astype({"SITE": str, "YEAR": int, "MONTH": int, "PRECIP": float, "TEMP": float})

    # synthesizing precipitation
    synth_precip = synthesize_precip(n, filtered_data, precip_dict, resolution, incomplete_years) 
    
    # ----------------------------
    # conditionally synthesizing targets
    # ----------------------------

    synth_pt = synthesize_pt_pairs(
        synth_precip,
        copula_dicts,
        filtered_data,
        resolution,
    )

    # ----------------------------
    # validation
    # ----------------------------
    if do_validation and False:
        print("Validating generated weather...")

        add_samples = (synthesize_kwargs["validation_samplesize_mult"] - 1) * n

        def build_sample(n_local):
            sp = synthesize_precip(
                n_local,
                filtered_data,
                precip_dict,
                resolution,
                incomp_years=[]
            )

            out = sp.copy()

            st = synthesize_pt_pairs(
                sp,
                copula_dicts,
                filtered_data,
                resolution,
            )

            for target_col in copula_dicts:
                out[target_col] = st[target_col].values

            return out

        if add_samples == 0:
            compare_synth_to_obs(
                validation_dirpath,
                validation_extension,
                synth_data,
                filtered_data
            )
        else:
            synth_extra = build_sample(add_samples)

            synth_extra["YEAR"] += synth_data["YEAR"].max() + 1

            combined = pd.concat([synth_data, synth_extra])
            combined.sort_values(by=["SITE", "YEAR", "MONTH"], inplace=True)
            combined.reset_index(drop=True, inplace=True)

            compare_synth_to_obs(
                validation_dirpath,
                validation_extension,
                combined,
                filtered_data
            )

    return synth_pt


def synthesize_precip(n_synth_years: int, data: pd.DataFrame, p_dict: dict, resolution: str, incomp_years: list[int]) -> np.array:
    """
    Manager function to synthesize precipitation

    Parameters
    ----------
    n_synth_years: int
        Number of years for the generator to synthesize weather for 
    data: pd.DataFrame
        Observed precipitation and temperature data
    p_dict: dict
        GMMHMM best-fitted model and corresponding parameters
    resolution: str
        Resolution of desired synthesized data
    incomp_years: list[int]
        List of years without the full set of months

    Returns
    -------
    synth_precip_monthly: pd.DataFrame
        Synthesized monthly precipitation
    """
    
    def precip_kNN_disaggregation(precip_log10annual_sample: np.array, 
                                  precip_obs: pd.DataFrame, 
                                  precip_log10annual_obs: pd.DataFrame) -> np.array:
        """
        Function to perform a k-NN disaggregation scheme for the log10(annual) precipitation 
        data specifically, adapted from ideas in Lall & Sharma (1996), 
        Apipattanavis et al. (2007), Nowak et al. (2010), and Quinn et al. (2020, supplmental). 
        This technique is non-parametric and therefore requires existing observations
        
        Parameters
        ----------
        precip_log10annual_sample: np.array
            Sample taken from the GMMHMM of the log10 annual precipitation data
        precip_obs: pd.DataFrame
            Observed precipitation data at the appropriate resolution
        precip_log10annual_obs: pd.DataFrame
            log10(annual)-transformed precipitation data as determined in
            the GMMHMM fit

        Returns
        -------
        disaggregated_sample: pd.DataFrame
            k-NN disaggregated monthly precipitation sample
        """

        # (0) convert to real-space from synth log-space
        n_months = 12
        synth_annual = 10.**precip_log10annual_sample

        # (1) create k, weights -- k recommended to be int(sqrt(n)), where n is the number of years
        k = round(np.sqrt(len(years)))
        w = np.array([(1 / j) for j in range(1, k+1)]) / sum([(1 / j) for j in range(1, k+1)])

        # (2) spatial averages for observations and sample
        obs_spatial_avg = np.nanmean(10.**precip_log10annual_obs.values, axis=1)
        synth_spatial_avg = np.nanmean(synth_annual, axis=1)

        # (3) choose one of the k closest observed years 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            year_obs_pair = np.reshape([[years[i], obs_spatial_avg[i]] for i in range(len(years))], newshape=(len(years), 2))
        kNN_selected_years = np.full(shape=precip_log10annual_sample.shape[0], fill_value=np.nan)# per synthesized year one of the k nearest years is assigned
        disaggregated_sample = np.full(shape=(precip_log10annual_sample.shape[0], n_months, len(sites)), fill_value=np.nan)
        for j, sum_synth_year in enumerate(synth_spatial_avg):
            # (4) calculate Manhattan distance (since 1D) between individual synthetic and all obs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                year_synth_dist = np.reshape([[year_obs_pair[i, 0], abs(sum_synth_year - year_obs_pair[i, 1])] for i in range(len(years))], newshape=(len(years), 2))
            sorted_year_dist = year_synth_dist[year_synth_dist[:, 1].argsort()]
            # (5) choose a year from the set using pre-determined weights
            kNN_selected_years[j] = rng.choice(sorted_year_dist[:k, 0], p=w)
        kNN_dict["precip"] = kNN_selected_years

        # (6) maintain temporal proportionality: synth_{month} = synth_{year} * hist_{month}/hist_{year}
        for i, kNN_selected_year in enumerate(kNN_selected_years):
            year_idx = precip_obs["YEAR"] == kNN_selected_year
            for s, site in enumerate(sites):
                site_idx = precip_obs["SITE"] == site
                if resolution == "monthly" or "DAY" not in precip_obs.columns:
                    kNN_selected_monthlies = precip_obs.loc[year_idx & site_idx, "PRECIP"].values
                else:
                    kNN_selected_monthlies = []
                    for m in range(1, n_months+1):
                        month_idx = precip_obs["MONTH"] == m
                        kNN_selected_monthlies.append(np.nansum(precip_obs.loc[month_idx & year_idx & site_idx, "PRECIP"].values))
                    kNN_selected_monthlies = np.array(kNN_selected_monthlies)
                disaggregated_sample[i, :, s] = synth_annual[i, s] * (kNN_selected_monthlies / sum(kNN_selected_monthlies))

        return disaggregated_sample
 
    rng = np.random.default_rng()
    filtered_annual_data = p_dict["log10_annual_precip"].drop(incomp_years)
    sites, years = sorted(set(data["SITE"].values)), sorted(set(filtered_annual_data.index.values))
    annual_sample = p_dict["model"].sample(n_synth_years)[0]
    precip_data = data[data.columns[:list(data.columns).index("PRECIP")+1]].copy()
    return precip_kNN_disaggregation(annual_sample, precip_data, filtered_annual_data) 


def synthesize_pt_pairs(synth_prcp: np.array, copula_dicts: dict, pt_df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    Manager function to conditionally synthesize temperature from 
    precipitation

    Parameters
    ----------
    synth_prcp: pd.DataFrame
        Synthesized precipitation at monthly resolution
    t_dict: dict
        Copula best-fitted, spatially-averaged models and corresponding parameters
    pt_df: pd.DataFrame
        Temporally-formatted observed precipitation and temperature data
    resolution: str
        Time resolution of the synthesized data

    Returns
    -------
    synth_monthly_df: pd.DataFrame
        Synthesized monthly precipitation, temperature minimum and maximum, irradiance, and ETP.
    """
    
    def conditionally_simulate_uT(poP: np.array, cop_list: list) -> np.array:
        """
        Conditionally simulate temperature pseudo-observations from
        synthesized precipitation pseudo-observations using the 
        appropriate copula family

        Parameters
        ----------
        poP: np.array
            Synthesized precipitation pseudo-observations
        cop_list: list
            List with the copula object and family name

        Returns
        -------
        poT: np.array
            Conditionally sampled temperature pseudo-observations
        """

        cop_obj, cop_name = cop_list[0], cop_list[1]
        if cop_name == "Independence":
            # v = d/du [C(u,v)] --> since C(u,v) = u*v, marginal v *is* the inverse of the conditional CDF
            poT = rng.random(size=len(poP))
        if cop_name == "Frank":
            # v = inverse of the conditional CDF -- c(v|u)^{-1} -- so the ppf of the copula given u
            y = rng.random(size=len(poP)) 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                try:
                    poT = cop_obj.percent_point(y, poP)
                except ValueError:
                    y = rng.random(size=len(poP))
                    poT = cop_obj.percent_point(y, poP)
        if cop_name == "Gaussian":
            # (1) conditional sampling starts with the Cholesky decomposition of the Gaussian parameter
            # (2) transform to normal distribution for poP, generate on normal distribution for y
            # (3) matrix multiply to get sample simulation given input
            # (4) temperature marginals are the cdf of the temperature half of the conditional sample
            A = np.tril(np.linalg.cholesky(cop_obj.correlation.values))
            normP = scipy.stats.norm.ppf(poP)
            y = scipy.stats.norm.ppf(rng.random(size=len(poP)))
            cond_samp = A @ np.array([normP, y])
            poT = scipy.stats.norm.cdf(cond_samp[1])

        return poT

    def temp_kNN_disaggregation(
        t_monthly_synth: np.array,
        ir_monthly_synth: np.array,
        pt_monthly_obs: pd.DataFrame,
        mnth: int,
    ):
        """
        Perform a k-NN disaggregation scheme for spatially averaged
        temperature data.

        k-NN selection uses temperature and irradiance jointly.
        """

        def get_observed_spatial_avg(pt_monthly_obs, target_col):
            obs_month_idx = pt_monthly_obs["MONTH"] == mnth

            obs_spatial_avg = []

            for year in years:
                obs_year_idx = pt_monthly_obs["YEAR"] == year

                obs_values = pt_monthly_obs.loc[
                    obs_month_idx & obs_year_idx,
                    target_col,
                ].values

                obs_spatial_avg.append(
                    np.inf
                    if all(np.isnan(obs_values))
                    else np.nanmean(obs_values)
                )

            return np.array(obs_spatial_avg)

        def select_knn_years(
            synth_temp,
            obs_temp,
            synth_ir=None,
            obs_ir=None,
        ):
            k = round(np.sqrt(len(years)))

            w = np.array([
                1 / j for j in range(1, k + 1)
            ])

            w /= w.sum()

            kNN_selected_years = np.full(
                shape=len(synth_temp),
                fill_value=np.nan,
            )

            for j in range(len(synth_temp)):

                if synth_ir is None:
                    distances = np.abs(
                        synth_temp[j] - obs_temp
                    )

                else:
                    temp_dist = np.abs(
                        synth_temp[j] - obs_temp
                    )

                    ir_dist = np.abs(
                        synth_ir[j] - obs_ir
                    )

                    distances = np.sqrt(
                        temp_dist**2 + ir_dist**2
                    )

                year_obs_pair = np.column_stack(
                    (years, distances)
                )

                sorted_year_dist = year_obs_pair[
                    year_obs_pair[:, 1].argsort()
                ]

                kNN_selected_years[j] = rng.choice(
                    sorted_year_dist[:k, 0],
                    p=w,
                )

            return kNN_selected_years

        def disaggregate(
            synth_spatial_avg,
            selected_years,
            target_col,
        ):
            obs_month_idx = (
                pt_monthly_obs["MONTH"] == mnth
            )

            kNN_spatial_avg = np.full(
                shape=len(synth_spatial_avg),
                fill_value=np.nan,
            )

            for i, selected_year in enumerate(selected_years):

                knn_year_idx = (
                    pt_monthly_obs["YEAR"] == selected_year
                )

                selected_values = pt_monthly_obs.loc[
                    knn_year_idx & obs_month_idx,
                    target_col,
                ].values

                kNN_spatial_avg[i] = np.nanmean(
                    selected_values
                )

            resids = (
                synth_spatial_avg - kNN_spatial_avg
            )

            disaggregated_sample = np.full(
                shape=(
                    len(synth_spatial_avg),
                    len(sites),
                ),
                fill_value=np.nan,
            )

            for i, selected_year in enumerate(selected_years):

                knn_year_idx = (
                    pt_monthly_obs["YEAR"] == selected_year
                )

                selected_values = pt_monthly_obs.loc[
                    knn_year_idx & obs_month_idx,
                    target_col,
                ].values

                noise = rng.normal(
                    loc=0.0,
                    scale=np.nanstd(resids),
                    size=1,
                )

                disaggregated_sample[i, :] = (
                    (
                        (synth_spatial_avg[i] + 273.15)
                        * (
                            (selected_values + 273.15)
                            / (
                                np.nanmean(selected_values)
                                + 273.15
                                + noise
                            )
                        )
                    )
                    - 273.15
                )

            return disaggregated_sample

        # =================================================
        # OBSERVED SPATIAL AVERAGES
        # =================================================

        obs_temp = get_observed_spatial_avg(
            pt_monthly_obs,
            "TEMP_MIN",
        )

        obs_ir = get_observed_spatial_avg(
            pt_monthly_obs,
            "IRRADIANCE",
        )

        # =================================================
        # SELECT HISTORICAL MONTHS
        # =================================================

        kNN_selected_years = select_knn_years(
            t_monthly_synth,
            obs_temp,
            ir_monthly_synth,
            obs_ir,
        )

        # =================================================
        # STORE SELECTED YEARS
        # =================================================

        if mnth not in kNN_dict:
            kNN_dict["temp"][mnth] = kNN_selected_years

        # =================================================
        # DISAGGREGATE
        # =================================================

        synth_temp = disaggregate(
            t_monthly_synth,
            kNN_selected_years,
            "TEMP_MIN",
        )

        synth_ir = disaggregate(
            ir_monthly_synth,
            kNN_selected_years,
            "IRRADIANCE",
        )

        return synth_temp, synth_ir, kNN_selected_years
    
    def daily_kNN_disaggregation(synth_month_df: pd.DataFrame, obs_daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Using previously discovered k-NN years and months, convert monthly
        WX data to daily data. This technique is non-parametric and therefore
        requires existing observations

        Parameters
        ----------
        synth_month_df: pd.DataFrame
            Synthetic precipitation, temperature, irradiance and ETP at monthly resolution
        obs_daily_df: pd.DataFrame
            Observed precipitation, temperature, irradiance and ETP at daily resolution

        Returns
        -------
        daily_df: pd.DataFrame
            k-NN disaggregated daily synthetic data
        """

        # average precip values per doy from observations
        doy_dict = {}
        for site in sorted(set(obs_daily_df["SITE"].values)):
            doy_site_idx = obs_daily_df["SITE"] == site
            doy_site_entry = obs_daily_df.loc[doy_site_idx]
            if site not in doy_dict: doy_dict[site] = {}
            for i in range(doy_site_entry.shape[0]):
                row_entry = doy_site_entry.iloc[i]
                doy = int(dt.datetime.strptime("{}-{}-{}".format(int(row_entry["YEAR"]), str(row_entry["MONTH"]).zfill(2), str(row_entry["DAY"]).zfill(2)), "%Y-%m-%d").strftime("%j"))
                if doy not in doy_dict[site]:
                    doy_dict[site][doy] = [row_entry["PRECIP"]]
                else:
                    doy_dict[site][doy].append(row_entry["PRECIP"])
            for doy in doy_dict[site]:
                doy_dict[site][doy] = float(np.nanmean(doy_dict[site][doy]))

        daily_dict = {}
        for site in sites:
            monthly_site_idx = synth_month_df["SITE"] == site
            monthly_site_entry = synth_month_df.loc[monthly_site_idx]
            daily_site_idx = obs_daily_df["SITE"] == site
            daily_site_entry = obs_daily_df.loc[daily_site_idx]

            for mnth in sorted(set(monthly_site_entry["MONTH"].values)):
                monthly_month_idx = monthly_site_entry["MONTH"] == mnth
                monthly_month_entry = monthly_site_entry.loc[monthly_month_idx]
                daily_month_idx = daily_site_entry["MONTH"] == mnth
                daily_month_entry = daily_site_entry.loc[daily_month_idx]

                for y in range(len(set(synth_month_df["YEAR"].values))):
                    monthly_year_idx = monthly_month_entry["YEAR"] == y+1
                    monthly_year_entry = monthly_month_entry.loc[monthly_year_idx]

                    daily_precip_year_idx = daily_month_entry["YEAR"] == kNN_dict["precip"][y]
                    daily_precip_year_entry = daily_month_entry.loc[daily_precip_year_idx]

                    daily_temp_year_idx = daily_month_entry["YEAR"] == kNN_dict["temp"][mnth][y]
                    daily_temp_year_entry = daily_month_entry.loc[daily_temp_year_idx]

                    sync_num_days = min([
                        daily_precip_year_entry.shape[0],
                        daily_temp_year_entry.shape[0]
                    ])

                    if np.nansum(daily_precip_year_entry["PRECIP"].values[:sync_num_days]) == 0:
                        sync_doys = []
                        for d in range(sync_num_days):
                            sync_doys.append(int(dt.datetime.strptime("{}-{}-{}".format(int(kNN_dict["precip"][y]), str(mnth).zfill(2), str(d+1).zfill(2)), "%Y-%m-%d").strftime("%j")))
                        sync_precips = [doy_dict[site][sync_doy] for sync_doy in sync_doys]
                    else:
                        sync_precips = daily_precip_year_entry["PRECIP"].values[:sync_num_days]

                    sync_precips = np.asarray(sync_precips, dtype=float)
                    sync_temp_min = np.asarray(daily_temp_year_entry["TEMP_MIN"].values[:sync_num_days], dtype=float)
                    sync_temp_max = np.asarray(daily_temp_year_entry["TEMP_MAX"].values[:sync_num_days], dtype=float)
                    sync_ir = np.asarray(daily_temp_year_entry["IRRADIANCE"].values[:sync_num_days], dtype=float)
                    sync_etp = np.asarray(daily_temp_year_entry["ETP"].values[:sync_num_days], dtype=float)

                    prcp = sync_precips * (monthly_year_entry["PRECIP"].values[0] / np.nansum(sync_precips))
                    temp_min = sync_temp_min + (monthly_year_entry["TEMP_MIN"].values[0] - np.nanmean(sync_temp_min))
                    temp_max = sync_temp_max + (monthly_year_entry["TEMP_MIN"].values[0] - np.nanmean(sync_temp_min))
                    ir = sync_ir * (monthly_year_entry["IRRADIANCE"].values[0] / np.nanmean(sync_ir))

                    # scale ETP according to the temperature-dependent saturation vapour pressure
                    def sat_vapour_pressure(temp):
                        return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))

                    etp = sync_etp * (sat_vapour_pressure(temp_min) / np.nanmean(sat_vapour_pressure(sync_temp_min)))

                    daily_dict.update({
                        (site, y+1, mnth, day+1): [
                            site, y+1, mnth, day+1,
                            prcp[day], temp_min[day], temp_max[day], ir[day], etp[day]
                        ]
                        for day in range(sync_num_days)
                    })

        daily_df = pd.DataFrame().from_dict(
            daily_dict,
            orient="index",
            columns=[
                "SITE",
                "YEAR",
                "MONTH",
                "DAY",
                "PRECIP",
                "TEMP_MIN",
                "TEMP_MAX",
                "IRRADIANCE",
                "ETP"
            ]
        )

        daily_df.sort_values(
            by=["SITE", "YEAR", "MONTH", "DAY"],
            inplace=True
        )

        daily_df.reset_index(drop=True, inplace=True)

        daily_df.astype({
            "SITE": str,
            "YEAR": int,
            "MONTH": int,
            "DAY": int,
            "PRECIP": float,
            "TEMP_MIN": float,
            "TEMP_MAX": float,
            "IRRADIANCE": float,
            "ETP": float
        })

        return daily_df

    def synthesize_target(uP, t_dict, target_col, month):
        # conditional simulation of the uT | uP --> coming from {d/d(uP) [C(uP, uT)]}^{-1}
        uT = conditionally_simulate_uT(uP, t_dict[month]["BestCopula"])

        # transform from marginals to residuals (using CDF^{-1}) to data (using AR fit params)
        resid_temp = t_dict[month][target_col + " Resid Dist"].ppf(uT)
        obs_mean, obs_std = np.nanmean(t_dict[month][target_col].astype(float)), np.nanstd(t_dict[month][target_col].astype(float))
        sa_synth_temp_approx = (resid_temp + t_dict[month][target_col + " ARFit"].params[0]) / (1. - t_dict[month][target_col + " ARFit"].params[1])
        approx_mean, approx_std = np.nanmean(sa_synth_temp_approx), np.nanstd(sa_synth_temp_approx)
        sa_synth_temp = (sa_synth_temp_approx - approx_mean)*(obs_std/approx_std) + obs_mean

        # (parametric) conditional temperature can sample values WAY too high or low
        # -- if this happens, resample the conditional temperatures until it doesn't happen
        obs_temp = t_dict[month][target_col].astype(float)
        obs_max_diff = np.abs(np.nanmax(obs_temp) - np.nanmin(obs_temp))
        while np.any(sa_synth_temp < np.nanmin(obs_temp) - obs_max_diff) or np.any(sa_synth_temp > np.nanmax(obs_temp) + obs_max_diff):
            uT = conditionally_simulate_uT(uP, t_dict[month]["BestCopula"])
            resid_temp = t_dict[month][target_col + " Resid Dist"].ppf(uT)
            sa_synth_temp_approx = (resid_temp + t_dict[month][target_col + " ARFit"].params[0]) / (1. - t_dict[month][target_col + " ARFit"].params[1])
            approx_mean, approx_std = np.nanmean(sa_synth_temp_approx), np.nanstd(sa_synth_temp_approx)
            sa_synth_temp = (sa_synth_temp_approx - approx_mean)*(obs_std/approx_std) + obs_mean

        return sa_synth_temp

    def aggregate_daily_to_monthly(pt_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
        if "DAY" not in pt_df.columns: return pt_df

        pt_monthly_df_dict = {}
        for site in sorted(set(pt_df["SITE"].values)):
            site_idx = pt_df["SITE"] == site
            for year in sorted(set(pt_df["YEAR"].values)):
                year_idx = pt_df["YEAR"] == year
                for month in sorted(set(pt_df["MONTH"].values)):
                    month_idx = pt_df["MONTH"] == month
                    prcps = pt_df.loc[site_idx & year_idx & month_idx, "PRECIP"].values
                    prcp_sum = np.nan if len(prcps) == 0 else np.nansum(prcps)
                    target_vals = []
                    for target_col in target_cols:
                        vals = pt_df.loc[site_idx & year_idx & month_idx, target_col].values
                        target_vals.append(np.nan if len(vals) == 0 else np.nanmean(vals))
                    pt_monthly_df_dict[(site, year, month)] = [site, year, month, prcp_sum, *target_vals]

        pt_monthly_df = pd.DataFrame().from_dict(
            pt_monthly_df_dict, orient="index",
            columns=["SITE", "YEAR", "MONTH", "PRECIP", *target_cols])
        pt_monthly_df.reset_index(drop=True, inplace=True)
        pt_monthly_df.astype({
            "SITE": str, "YEAR": int, "MONTH": int,
            "PRECIP": float, **{col: float for col in target_cols}})
        return pt_monthly_df
    
    t_dict = copula_dicts["TEMP_MIN"]

    rng = np.random.default_rng()
    sites = sorted(set(pt_df["SITE"].values))
    years = [y for y in range(min(pt_df["YEAR"].values), max(pt_df["YEAR"].values)+1)]
    month_names, month_vals = list(t_dict.keys()), [m+1 for m in range(len(t_dict.keys()))]
    n_synth_years, n_months, n_sites = synth_prcp.shape
    synth_monthly_df = pd.DataFrame(columns=["SITE", "YEAR", "MONTH", "PRECIP", "TEMP_MIN", "TEMP_MAX", "IRRADIANCE", "ETP"])
    synth_monthly_df["SITE"] = np.repeat(sites, n_synth_years * n_months)
    synth_monthly_df["YEAR"] = list(np.repeat([y+1 for y in range(n_synth_years)], n_months)) * n_sites
    synth_monthly_df["MONTH"] = month_vals * (n_synth_years * n_sites)

    pt_monthly_df = aggregate_daily_to_monthly(pt_df, ["TEMP_MIN", "IRRADIANCE"])

    # calculate precipitation pseudo-observations
    uP_dict = {}
    for m, month in enumerate(month_names):
        # spatially average synth precip for the month
        sa_synth_prcp = synth_prcp[:, m, :].mean(axis=1)
        
        # transform the synthetic preciptation to residuals using the ARfit used in the copulas
        nP = len(sa_synth_prcp)
        ar1_synth_prcp_fit = t_dict[month]["PRECIP ARFit"].apply(sa_synth_prcp)
        full_ar1_prcp = np.array([np.nanmean(ar1_synth_prcp_fit.fittedvalues), *ar1_synth_prcp_fit.fittedvalues])
        resid_prcp = sa_synth_prcp - full_ar1_prcp

        # transform into uniform marginals
        uP_dict[month] = stats.rankdata(resid_prcp, method="average") / (nP+1)

    # First loop: conditionally synthesize all target variables
    synth_targets = {}

    for target_col in list(copula_dicts.keys()):
        t_dict = copula_dicts[target_col]
        synth_targets[target_col] = {}

        for m, month in enumerate(month_names):
            uP = uP_dict[month]
            synth_targets[target_col][month] = synthesize_target(uP, t_dict, target_col, month)


    # Second loop: jointly disaggregate the synthesized target variables
    for m, month in enumerate(month_names):
        month_idx = synth_monthly_df["MONTH"] == month_vals[m]

        sa_synth_temp = synth_targets["TEMP_MIN"][month]
        sa_synth_ir = synth_targets["IRRADIANCE"][month]

        # take the spatially-averaged temperature and irradiance and disaggregate jointly
        synth_temp, synth_ir, _ = temp_kNN_disaggregation(
            sa_synth_temp,
            sa_synth_ir,
            pt_monthly_df,
            month_vals[m]
        )

        # assign to dataframe
        for s, site in enumerate(sites):
            site_idx = synth_monthly_df["SITE"] == site
            synth_monthly_df.loc[month_idx & site_idx, "PRECIP"] = synth_prcp[:, m, s]
            synth_monthly_df.loc[month_idx & site_idx, "TEMP_MIN"] = synth_temp[:, s]
            synth_monthly_df.loc[month_idx & site_idx, "IRRADIANCE"] = synth_ir[:, s]
    
    if resolution == "monthly" or (resolution == "daily" and "DAY" not in pt_df.columns):
        if resolution == "daily" and "DAY" not in pt_df.columns:
            warnings.warn("Input dataset at monthly resolution cannot be disaggregated to daily! Returning monthly...", UserWarning)
        return synth_monthly_df
    else:
        # disaggregate to daily values using the closest historical month and scale to the synthesized monthly values
        return daily_kNN_disaggregation(synth_monthly_df, pt_df)

