from typing import List, Tuple, Dict 
import numpy as np
import pandas as pd
import datetime as dt
import random
from hmmlearn.hmm import GMMHMM
import warnings
from statsmodels.tsa.ar_model import AutoReg
from copulas import univariate, bivariate, multivariate
import scipy
import copulae
from statsmodels.tools import eval_measures
import logging

from .make_figures import *


def fit_data(data: pd.DataFrame, 
             resolution: str,
             validation: bool,
             dirpath: str,
             fit_kwargs: dict) -> list[dict, dict]:
    """
    Managing function that fits the raw climate/weather data as a reformatted DataFrame,
    with statistics and sampling schemes for precipitation, additional parameters
    (i.e. temperature), and the relationship between the two

    Parameters
    ----------
    data: pd.DataFrame
        Temporally reformatted data to be used for the fitting
    resolution: str
        The temporal resolution of the input data. Can be 'monthly' or 'daily'
    validation: bool
        Flag for producing figures to validate each step of the generator
    dirpath: str
        Path for where to save the validation figures
    fit_kwargs: dict
        Dictionary with the fit keyword arguments

    Returns
    -------
    precip_fit_dict: dict
        Dictionary containing statistical information related to fitting of precipitation data
    copulaetemp_fit_dict: dict
        Dictionary containing statistical information related to conditional fitting of temperature 
        data through copulae
    """
   
    # fit kwargs
    default_fit_kwargs = {"gmhmm_min_states": 1,
                          "gmhmm_max_states": 4,
                          "gmhmm_states": 0,
                          "max_convergence_attempts": 50,
                          "ar_lag": 1,
                          "copula_families": ["Independence", "Frank", "Gaussian"],
                          "figure_extension": "svg",
                          "validation_figures": ["precip", "copula"],
                          "fit_verbose": True}
    if not fit_kwargs: 
        fit_kwargs = default_fit_kwargs
    else:
        for k in default_fit_kwargs:
            if k not in fit_kwargs:
                fit_kwargs[k] = default_fit_kwargs[k] 
    
    # validation
    global do_validation, validation_dirpath, validation_extension, validation_figures, fit_verbose, max_convergence_attempts
    do_validation, validation_dirpath, validation_extension, validation_figures = validation, dirpath, fit_kwargs["figure_extension"], fit_kwargs["validation_figures"]
    fit_verbose = fit_kwargs["fit_verbose"]
    max_convergence_attempts = fit_kwargs["max_convergence_attempts"]

    # precip
    precip_col_idx = list(data.columns).index("PRECIP")
    precip_fit_dict = fit_precip(data[data.columns[:precip_col_idx+1]].copy(), 
                                 resolution,
                                 fit_kwargs["gmhmm_min_states"],
                                 fit_kwargs["gmhmm_max_states"],
                                 fit_kwargs["gmhmm_states"])

    # copulae/temp
    copulaetemp_fit_dict = fit_copulae(data[data.columns[:precip_col_idx+1].append(pd.Index(["TEMP_MIN"]))].copy(), 
                                       resolution, 
                                       precip_fit_dict["log10_annual_precip"].index.values,
                                       fit_kwargs["ar_lag"], 
                                       fit_kwargs["copula_families"],
                                       target_col="TEMP_MIN")
    
    copulaeirrad_fit_dict = fit_copulae(data[data.columns[:precip_col_idx+1].append(pd.Index(["IRRADIANCE"]))].copy(), 
                                    resolution, 
                                    precip_fit_dict["log10_annual_precip"].index.values,
                                    fit_kwargs["ar_lag"], 
                                    fit_kwargs["copula_families"],
                                    target_col="IRRADIANCE")
    
    # validation for fits
    if do_validation:
        validate_pt_fits(validation_dirpath, validation_extension, data, precip_fit_dict, copulaetemp_fit_dict, validation_figures)


    return (
        precip_fit_dict,
        copulaetemp_fit_dict,
        copulaeirrad_fit_dict,
    )


def fit_precip(data: pd.DataFrame, resolution: str, min_states: int, max_states: int, fixed_states: int) -> dict:
    """
    Function that fits and validates the precipitation data. Precipitation is transformed
    to a log-scale, annualized (summed), and fit to a Gaussian mixture Hidden 
    Markov model (GMHMM)
    
    Parameters
    ----------
    data: pd.DataFrame
        Temporal reformat of ``raw_data`` where each year, month, day have their own column
        in the DataFrame
    resolution: str
        The temporal resolution of the input data. Can be 'monthly' or 'daily' 
    min_states: int
        The minimum number of hidden states to try fitting 
    max_states: int
        The maximum number of hidden states to try fitting. More than ~6 tends
        to perform poorly in terms of best fit and length of computation 
    fixed_states: int
        Do not try fitting for best number of hidden states and only use this value
    
    Returns
    -------
    precip_fit_dict: dict
        Dictionary containing statistical information related to fitting of precipitation data
    """

    def gmmhmm_state_num_estimator(transformed_df: pd.DataFrame, lengths: list[int], min_states: int, max_states: int, iterations: int = 10) -> int:
        """
        Function to programmatically determine the best-fitting number of states 
        for the Gaussian mixture model hidden Markov model
        
        Parameters
        ----------
        transformed_df: pd.DataFrame
            DataFrame for the log10-transformed annualized precipitation values, with
            associated sites and years organized as a DataFrame where indices are
            years and columns are sites
        lengths: list[int]
            Length of each sequence of consecutive years in the data
        min_states: int
            The minimum number of hidden states to try fitting. Default: 1 
        max_states: int
            The maximum number of hidden states to try fitting. More than ~6 tends
            to perform poorly in terms of best fit and length of computation. Default: 5 
        iterations: int, fixed
            The number of attempts to find a best-fitting model for each unique
            number of states -- this is necessary because the convergence EM method can 
            fall into local optima. From all iterations, the one with the highest 
            log-likelihood is used. Using: 10

        Returns
        -------
        best_num_states: int
            Best-fitting model's number of states, as determined by Bayesian Information
            Criterion (BIC). Log-likelihood and Akaike Information Criterion (AIC) are 
            also used, but log-likelihood is monotonically increasing with number of states
            and (from experimentation) AIC does not penalize additional states strongly enough. 
        """

        n_stations = len(transformed_df.columns)
        models, seeds, AICs, BICs, LLs = [], [], [], [], []
        max_attempts = max_convergence_attempts
        for num_states in range(min_states, max_states + 1):
            best_model, best_LL, best_seed = None, None, None
            for _ in range(iterations):
                positive_definite, temp_seed, temp_model = False, None, None
                attempts = 0
                while (not positive_definite) and (attempts < max_attempts):
                    # get the random state
                    tempSeed = random.getstate()

                    # define the parameters for the model
                    temp_model_inst = GMMHMM(n_components=num_states, n_iter=1000, covariance_type="full", init_params="cmw", verbose=False)
                    temp_model_inst.startprob_ = np.full(shape=num_states, fill_value=1./num_states)
                    temp_model_inst.transmat_ = np.full(shape=(num_states, num_states), fill_value=1./num_states)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        temp_model = temp_model_inst.fit(transformed_df.values, lengths=lengths)

                    # get, reshape the covariance matrices
                    temp_covars = temp_model.covars_.reshape((num_states, n_stations, n_stations))

                    # check that each state has a covariance matrix that is positive definite, meaning:
                    # -- it's symmetric
                    symmetric_check = all([np.allclose(temp_covars[i].T, temp_covars[i]) for i in range(num_states)])
                    # -- it's eigenvalues are all positive
                    eig_check = all([(np.linalg.eigvalsh(temp_covars[i]) > 0).all() for i in range(num_states)])

                    # state if the covariance matrix is positive definite or not
                    positive_definite = symmetric_check and eig_check
                    attempts += 1

                # calculate the loglikelihood of this model, save best
                if attempts < max_attempts: 
                    temp_score = temp_model.score(transformed_df.values, lengths=lengths)
                    if not best_LL or temp_score > best_LL:
                        best_LL = temp_score
                        best_model = temp_model
                        best_seed = temp_seed
            # save the best metrics for each number of states
            if attempts < max_attempts:
                models.append(best_model)
                seeds.append(best_seed)
                LLs.append(best_LL)
                AICs.append(best_model.aic(transformed_df.values, lengths=lengths))
                BICs.append(best_model.bic(transformed_df.values, lengths=lengths))
                if fit_verbose:
                    print("Positive definite covariance matrix for GMHMM fit found for {} state(s)!".format(num_states))
            else:
                if fit_verbose:
                    print("Positive definite covariance matrix for GMHMM fit cannot be found for {} states...".format(num_states))
        model = models[np.argmin(BICs)]
        
        # validate if prompted
        if do_validation and ("precip" in validation_figures):
            validate_gmhmm_states(validation_dirpath, validation_extension, min_states, max_states, LLs, AICs, BICs)
        
        return model.n_components
    

    def conceptual_reorder(model_means: np.array, model_covars: np.array, model_transmat: np.array) -> tuple[np.array, np.array, np.array]:
        """
        Function to conceptually reorder the discovered ``n`` states, with the ``0``th index
        being the driest state and the (``n-1``)th index being the wettest state

        Parameters
        ----------
        model_means: np.array
            Means discovered by the GMMHMM, ordered as discovered
        model_covars: np.array
            Covariance matrix discovered by the GMMHMM, ordered as discovered
        model_transmat: np.array
            Transition matrix discovered by the GMMHMM, ordered as discovered
        
        Returns
        -------
        reordered_means: np.array
            Means reordered so that the driest state is the first index and the 
            wettest is the last
        reordered_covars: np.array
            Means reordered so that the driest state is the first index and the 
            wettest is the last
        reordered_transmat: np.array
            Means reordered so that the driest state is the first index and the 
            wettest is the last
        """
        
        # reorder indices, return if already in order
        conceptual_order_indices = np.argsort([np.sum(model_means[i]) for i in range(len(model_means))])
        if np.all(np.diff(conceptual_order_indices) >= 0):
            return model_means, model_covars, model_transmat

        # reorder means, covariances
        reordered_means = model_means[conceptual_order_indices, :]
        reordered_covars = model_covars[conceptual_order_indices, :]

        # reorder transmat
        reordered_transmat = np.full(shape=model_transmat.shape, fill_value=np.nan)
        for i in range(model_transmat.shape[0]):
            reordered_transmat[conceptual_order_indices[i], :] = model_transmat[i, conceptual_order_indices]

        # return the reordered matrices
        return reordered_means, reordered_covars, reordered_transmat
    
    
    logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
    # annualize precipitation, log10 transformation, format so index=years, column=sites, cell=precip
    sites, years = sorted(set(data["SITE"].values)), sorted(set(data["YEAR"].values))
    transformed_data = pd.DataFrame(index=years, columns=sites)
    for site in sites:
        site_idx = data["SITE"] == site
        site_entry = data[site_idx]
        site_years = sorted(set(site_entry["YEAR"].values))
        for year in years:
            year_idx = site_entry["YEAR"] == year
            year_entry = site_entry[year_idx]
            if year_entry.empty: continue
            transformed_data.at[year, site] = np.log10(np.sum(year_entry["PRECIP"].values))
    transformed_data.astype({col: float for col in sites})
    transformed_data.dropna(inplace=True)

    # checking if there are any years missing from the data, making sequences of existing years
    seq_lengths, l = [], 0
    for year in years:
        if year in list(transformed_data.index):
            l += 1
        else:
            seq_lengths.append(l)
            l = 0
    seq_lengths.append(l)
    seq_lengths = [l for l in seq_lengths if l != 0]

    # determine best-fitting number of states for the GMHMM
    if fixed_states > 0:
        num_states = fixed_states
    else:
        num_states = gmmhmm_state_num_estimator(transformed_data, seq_lengths, min_states=min_states, max_states=max_states)
    
    # use these num_states to fit the GMHMM
    best_model, best_LL, best_seed = None, None, None
    for _ in range(20):
        positive_definite, temp_seed, temp_model = False, None, None
        while not positive_definite:
            tempSeed = random.getstate()
            temp_model_inst = GMMHMM(n_components=num_states, n_iter=1000, covariance_type="full", init_params="cmw", verbose=False)
            temp_model_inst.startprob_ = np.full(shape=num_states, fill_value=1./num_states)
            temp_model_inst.transmat_ = np.full(shape=(num_states, num_states), fill_value=1./num_states)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                temp_model = temp_model_inst.fit(transformed_data.values, lengths=seq_lengths)
            temp_covars = temp_model.covars_.reshape((num_states, len(sites), len(sites)))
            symmetric_check = all([np.allclose(temp_covars[i].T, temp_covars[i]) for i in range(num_states)])
            eig_check = all([(np.linalg.eigvalsh(temp_covars[i]) > 0).all() for i in range(num_states)])
            positive_definite = symmetric_check and eig_check

        temp_score = temp_model.score(transformed_data.values, lengths=seq_lengths)
        if not best_LL or temp_score > best_LL:
            best_LL = temp_score
            best_model = temp_model
            best_seed = temp_seed
    seed, model = best_seed, best_model
    model.means_, model.covars_, model.transmat_ = conceptual_reorder(model.means_, model.covars_, model.transmat_)
 
    # predict the hidden state for each datapoint
    hidden_states = model.predict(transformed_data.values, lengths=seq_lengths)

    # extract means, covariance, standard deviation from model
    means = model.means_.reshape((model.n_components, len(sites)))
    masked_covars = model.covars_.copy()
    masked_covars[masked_covars < 0] = np.nan
    stds = np.sqrt(masked_covars).reshape((model.n_components, len(sites), len(sites)))
    stds = np.array([[stds[n][i][i] for i in range(len(sites))] for n in range(num_states)])

    # null hypothesis testing
    pvalues = []
    for state in range(num_states):
        state_data = transformed_data.values[hidden_states == state]
        state_means, state_stds = means[state], stds[state]
        state_pvalues = []
        for s, site in enumerate(sites):
            site_data, mean, std = state_data[:, s], state_means[s], state_stds[s]
            site_pvalues = []
            for test in ["ad", "cvm", "ks"]:
                site_pvalues.append(round(float(scipy.stats.goodness_of_fit(scipy.stats.norm, site_data, known_params={"loc": mean, "scale": std}, statistic=test).pvalue), 4))
            site_pvalues = tuple(site_pvalues)
            state_pvalues.append(site_pvalues)
        pvalues.append(state_pvalues)

    # return the model and associated statistics as a dictionary
    precip_fit_dict = {}
    precip_fit_dict["log10_annual_precip"] = transformed_data
    precip_fit_dict["seq_lengths"] = seq_lengths
    precip_fit_dict["num_gmmhmm_states"] = num_states
    precip_fit_dict["seed"] = seed
    precip_fit_dict["model"] = model
    precip_fit_dict["means"] = means
    precip_fit_dict["covars"] = masked_covars
    precip_fit_dict["stds"] = stds
    precip_fit_dict["hidden_states"] = hidden_states
    precip_fit_dict["t_probs"] = model.transmat_
    precip_fit_dict["pvalues"] = pvalues
    return precip_fit_dict


def fit_copulae(data: pd.DataFrame, resolution: str, precip_fit_years: list[int], 
                ar_lag: int, copula_families: list[str], target_col="TEMP_MIN") -> dict:
    """
    Function that fits and validates the temperature data through fitting of 
    hydroclimatic copulae. Precipitation and temperature data are both first assessed
    by their Kendall and Spearman correlations. Next, they are passed through an AR(n) 
    filter -- default AR(1) -- to calculate residuals. These residuals can be visually 
    assessed for stationarity and their dependence structure (Kendall plots). 
    Pseudo-observations are created from the residuals, and then copulae of different 
    families are fit to the pseudo-observations (Independence, Frank, and Gaussian
    copula families are available in the current version).
    
    Parameters
    ----------
    data: pd.DataFrame
        Temporal reformat of ``raw_data`` where each year, month, day have their own column
        in the DataFrame
    resolution: str
        The temporal resolution of the input data. Can be 'monthly' or 'daily' 
    precip_fit_years: list[int]
        Years that precipitation was fit to
    ar_lag: int
        The time lag to consider in the AR fit step
    copula_families: list[str]
        The type of copula to consider when choosing a best-fitting family

    Returns
    -------
    pt_dict: dict
        Dictionary containing statistical information related to fitting of copulae/temp data
    """
     
    def investigate_autocorrelation(data_dict: dict, lag: int) -> dict:
        """
        Apply an autoregressive fit to the precipitation and temperature
        data to (1) find the fit, and; (2) calculate residuals

        Parameters
        ----------
        data_dict: dict
            Dictionary with the month-separated, spatially-averaged precipitation 
            and temperature data
        lag: int
            Lag to apply in the AR fit

        Returns
        -------
        data_dict: dict
            Same dictionary as above, with new keys for the fitted data and residuals
        """
        
        for month in data_dict.keys():
            p, t = data_dict[month]["PRECIP"], pt_dict[month][target_col]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                data_dict[month]["PRECIP ARFit"] = AutoReg(p, lags=[lag]).fit()
                data_dict[month][target_col + " ARFit"] = AutoReg(t, lags=[lag]).fit()

            # find the underlying univariate distribution of the residuals
            p_univariate, t_univariate = univariate.Univariate(), univariate.Univariate()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                p_univariate.fit(np.array(data_dict[month]["PRECIP ARFit"].resid, dtype=float))
                t_univariate.fit(np.array(data_dict[month][target_col + " ARFit"].resid, dtype=float))
            data_dict[month]["PRECIP Resid Dist"] = p_univariate
            data_dict[month][target_col + " Resid Dist"] = t_univariate
        
        # validate ACF of the raw data and residual autoregression/autocorrelation
        if do_validation and ("copula" in validation_figures):
            validate_pt_acf(validation_dirpath, validation_extension, data_dict, lag, target_col)
        
        return data_dict

    
    def fit_copula_families(data_dict: dict, families: list[str]) -> dict:
        """
        Fit copulae to the residuals and rank the best fitting copula
        (per month) according to the families outlined in ``copula_families``

        Parameters
        ----------
        data_dict: dict
            Dictionary with the month-separated, spatially-averaged precipitation 
            and temperature data
        copula_families: list[str]
            Families to consider in the copula fitting process

        Returns
        -------
        data_dict: dict
            Same dictionary as above, with new keys for fitted copula
        """
        
        def CalculateEmpiricalCopulaCDFatPoint(pseudo_observations: np.array, point: float) -> np.array:
            n = pseudo_observations.shape[0]
            vec_Cn = np.full(shape=n, fill_value=0.)
            for k in range(n):
                U1, U2 = pseudo_observations[k, 0], pseudo_observations[k, 1]
                if U1 <= point[0] and U2 <= point[1]:
                    vec_Cn[k] = 1.
            return np.sum(vec_Cn) / n

        def BootstrapCopulaCVMandKS(pseudo_observations: np.array, theory_copula, fam, from_df: bool = False) -> list[np.array, np.array]:
            n, d, N = pseudo_observations.shape[0], pseudo_observations.shape[1], 200
            if from_df:
                column_names = theory_copula.to_dict()["columns"]
             
            # Cramér Von-Mises (S_n), Kolmogorov-Smirnv (T_n) statistics
            Sn_elements = np.full(shape=n, fill_value=np.nan)
            Tn_elements = np.full(shape=n, fill_value=np.nan)
            for k in range(n):
                po = pseudo_observations[k, :]
                C_n = CalculateEmpiricalCopulaCDFatPoint(pseudo_observations, po)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    Bstar_m = theory_copula.cdf(pd.DataFrame(data={column_names[0]: [po[0]], column_names[1]: [po[1]]})) if from_df else theory_copula.cdf(np.atleast_2d(po))
                Bstar_m = Bstar_m[0] if type(Bstar_m) in [list, np.ndarray] else Bstar_m
                Sn_elements[k] = (C_n - Bstar_m)**2.
                Tn_elements[k] = np.abs(C_n  - Bstar_m)
            S_n = np.sum(Sn_elements)
            T_n = np.max(Tn_elements)
 
            # bootstrapping Cramér Von-Mises (S_n), Kolmogorov-Smirnvo (T_n) p-values
            bootstrap_Sn, bootstrap_Tn = [], []
            for _ in range(N):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    Ystar_k = theory_copula.random(n) if fam == "independent" else theory_copula.sample(n)
                Ystar_k = Ystar_k.values if from_df else Ystar_k
                Rstar_k = np.zeros_like(Ystar_k) 
                for j in range(d):
                    Rstar_k[:, j] = scipy.stats.rankdata(Ystar_k[:, j], method="ordinal")
                Ustar_k = Rstar_k / (n + 1)
                if fam == "independent":
                    bootstrap_cop = copulae.IndepCopula()
                if fam == "frank":
                    bootstrap_cop = bivariate.Frank()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=DeprecationWarning)
                        bootstrap_cop.fit(Ustar_k)
                if fam == "gaussian":
                    bootstrap_cop = multivariate.GaussianMultivariate(distribution={"uP": univariate.UniformUnivariate(),
                                                                                    "uT": univariate.UniformUnivariate()})
                    bootstrap_obs_df = pd.DataFrame(data=Ustar_k, columns=["uP", "uT"])
                    bootstrap_cop.fit(bootstrap_obs_df)
                bootstrap_sn, bootstrap_tn = [], []
                for k in range(n):
                    po = Ustar_k[k, :]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=DeprecationWarning)
                        Cstar_k = CalculateEmpiricalCopulaCDFatPoint(Ustar_k, po)
                        Bstar_k = bootstrap_cop.cdf(pd.DataFrame(data={column_names[0]: [po[0]], column_names[1]: [po[1]]})) if from_df else bootstrap_cop.cdf(np.atleast_2d(po))
                    Bstar_k = Bstar_k[0] if type(Bstar_k) in [list, np.ndarray] else Bstar_k
                    bootstrap_sn.append((Cstar_k - Bstar_k)**2.)
                    bootstrap_tn.append(np.abs(Cstar_k  - Bstar_k))
                bootstrap_Sn.append(np.sum(bootstrap_sn))
                bootstrap_Tn.append(np.max(bootstrap_tn))
            pS_n = float(np.nansum(np.array(bootstrap_Sn) > S_n) / N)
            pT_n = float(np.nansum(np.array(bootstrap_Tn) > T_n) / N)

            return S_n, T_n, (pS_n, pT_n)

        def FindBestCopula(copula_df: pd.DataFrame):
            aic_sorted, sn_sorted, tn_sorted = copula_df.copy().sort_values(by=["AIC"]), copula_df.copy().sort_values(by=["S_n"]), copula_df.copy().sort_values(by=["T_n"])

            # choosing the best copula
            winning_families = [aic_sorted.index.values[0], sn_sorted.index.values[0], tn_sorted.index.values[0]]
            n_best_families = len(set(winning_families))
            if n_best_families == 1 or n_best_families == 3:
                # -- if all three metrics are the lowest for a single family, use that one
                # -- if each family claims one lowest metric, just use AIC
                # ---- if AIC is at least 2 less each other family, that's a better model
                return [copula_df.at[aic_sorted.index.values[0], "Copula"], aic_sorted.index.values[0]]
            else:
                # pick the family where both S_n and T_n are winning if true, otherwise pick the winning AIC family
                if len({sn_sorted.index.values[0], tn_sorted.index.values[0]}) == 1:
                    return [copula_df.at[winning_families[1], "Copula"], winning_families[1]]
                else:
                    return [copula_df.at[winning_families[0], "Copula"], winning_families[0]]


        # include the pseudo-observations in the dictionary
        for month in data_dict.keys():
            nP, nT = len(data_dict[month]["PRECIP ARFit"].resid), len(data_dict[month][target_col + " ARFit"].resid)
            p_resids, t_resids = data_dict[month]["PRECIP ARFit"].resid, data_dict[month][target_col + " ARFit"].resid
            mask = ~(np.isnan(p_resids) | np.isnan(t_resids))
            data_dict[month]["PRECIP pObs"] = scipy.stats.rankdata(p_resids[mask], method="average") / (nP+1)
            data_dict[month][target_col + " pObs"] = scipy.stats.rankdata(t_resids[mask], method="average") / (nT+1)

        copula_fit_dict = {month: pd.DataFrame(data={"Copula": [None] * len(families),
                                                     "params": [np.nan] * len(families),
                                                     "AIC": [np.inf] * len(families),
                                                     "S_n": [[]] * len(families),
                                                     "T_n": [[]] * len(families),
                                                     "(S_n, T_n) P-Value": [[]] * len(families)},
                                               index=families) for month in data_dict}
        for month in data_dict:
            pseudo_obs = np.array([data_dict[month]["PRECIP pObs"], data_dict[month][target_col + " pObs"]]).T

            if "Independence" in families:
                iCop = copulae.IndepCopula()
                copula_fit_dict[month].at["Independence", "Copula"] = iCop
                copula_fit_dict[month].at["Independence", "params"] = np.nan
                copula_fit_dict[month].at["Independence", "AIC"] = eval_measures.aic(llf=iCop.log_lik(pseudo_obs),
                                                                                     nobs=pseudo_obs.size, df_modelwc=np.array([]).size)
                iS_n, iT_n, iPVals = BootstrapCopulaCVMandKS(pseudo_obs, iCop, "independent")
                copula_fit_dict[month].at["Independence", "S_n"] = iS_n
                copula_fit_dict[month].at["Independence", "T_n"] = iT_n
                copula_fit_dict[month].at["Independence", "(S_n, T_n) P-Value"] = iPVals

            if "Frank" in families:
                fCop = bivariate.Frank()
                skip_frank = False
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    try:
                        fCop.fit(pseudo_obs)
                    except ValueError:
                        skip_frank = True
                        print(f"[frank copula skip] {target_col} in {month}: constant/degenerate data")
                if not skip_frank:
                    copula_fit_dict[month].at["Frank", "Copula"] = fCop
                    copula_fit_dict[month].at["Frank", "params"] = fCop.theta
                    copula_fit_dict[month].at["Frank", "AIC"] = eval_measures.aic(llf=np.sum(fCop.log_probability_density(pseudo_obs)),
                                                                                nobs=pseudo_obs.size, df_modelwc=np.array(fCop.theta).size)
                    fS_n, fT_n, fPVals = BootstrapCopulaCVMandKS(pseudo_obs, fCop, "frank")
                    copula_fit_dict[month].at["Frank", "S_n"] = fS_n
                    copula_fit_dict[month].at["Frank", "T_n"] = fT_n
                    copula_fit_dict[month].at["Frank", "(S_n, T_n) P-Value"] = fPVals

            if "Gaussian" in families:
                pseudo_obs_df = pd.DataFrame(data=pseudo_obs, columns=["uP", "uT"])
                gCop = multivariate.GaussianMultivariate(distribution={"uP": univariate.UniformUnivariate(),
                                                                       "uT": univariate.UniformUnivariate()})
                gCop.fit(pseudo_obs_df)
                copula_fit_dict[month].at["Gaussian", "Copula"] = gCop
                copula_fit_dict[month].at["Gaussian", "params"] = gCop.correlation["uP"]["uT"]
                gCopulae = copulae.GaussianCopula()
                gCopulae.params = gCop.correlation["uP"]["uT"]
                copula_fit_dict[month].at["Gaussian", "AIC"] = eval_measures.aic(llf=gCopulae.log_lik(data=pseudo_obs_df.values, to_pobs=False),
                                                                                 nobs=pseudo_obs.size, df_modelwc=np.array(gCop.correlation["uP"]["uT"]).size)
                gS_n, gT_n, gPVals = BootstrapCopulaCVMandKS(pseudo_obs, gCop, "gaussian", from_df=True)
                copula_fit_dict[month].at["Gaussian", "S_n"] = gS_n
                copula_fit_dict[month].at["Gaussian", "T_n"] = gT_n
                copula_fit_dict[month].at["Gaussian", "(S_n, T_n) P-Value"] = gPVals

            # add the copula families dictionary to the ptDict, for conciseness
            data_dict[month]["CopulaDF"] = copula_fit_dict[month]
            data_dict[month]["BestCopula"] = FindBestCopula(copula_fit_dict[month])

        return data_dict
    
    
    sites = sorted(set(data["SITE"].values))
    years = precip_fit_years
    full_years = [y for y in range(np.nanmin(years), np.nanmax(years)+1)]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # aggregate the data into monthly data (if not already)
    if "DAY" in data.columns:
        monthed_dict = {}
        for site in sites:
            site_idx = data["SITE"] == site
            for year in years:
                year_idx = data["YEAR"] == year
                for i in range(len(month_names)):
                    month_idx = data["MONTH"] == i+1
                    month_entry = data.loc[site_idx & year_idx & month_idx]
                    if month_entry.empty:
                        monthly_precip, monthly_temp = np.nan, np.nan
                    else:
                        monthly_precip = np.nansum(month_entry["PRECIP"].values)
                        monthly_temp = np.nanmean(month_entry[target_col].values)
                    monthed_dict[(site, year, month_names[i])] = [site, year, month_names[i], monthly_precip, monthly_temp]
        pt_df = pd.DataFrame().from_dict(monthed_dict, orient="index", columns=["SITE", "YEAR", "MONTH", "PRECIP", target_col])
        pt_df.reset_index(drop=True, inplace=True)
    else:
        pt_df = data.copy()
        pt_df["MONTH"] = [month_names[i-1] for i in pt_df["MONTH"].values]

    # validate exploration of correlation between precipitation and temperature if prompted
    if do_validation and ("copula" in validation_figures):
        validate_explore_pt_dependence(validation_dirpath, validation_extension, pt_df, years, target_col)

    # establishing a dictionary, filling missing values from the group average
    pt_dict = {month: {"PRECIP": [], target_col: []} for month in month_names}
    for month in pt_dict:
        month_idx = pt_df["MONTH"] == month
        p, t = [], []
        for year in full_years:
            year_idx = pt_df["YEAR"] == year
            spatial_entry = pt_df.loc[month_idx & year_idx]
            p.append(np.nan if (spatial_entry.empty) or (np.all(np.isnan(spatial_entry["PRECIP"].values))) else np.nanmean(spatial_entry["PRECIP"].astype(float).values))
            t.append(np.nan if (spatial_entry.empty) or (np.all(np.isnan(spatial_entry[target_col].values))) else np.nanmean(spatial_entry[target_col].astype(float).values))
        p, t = np.array(p), np.array(t)
        p[np.isnan(p)], t[np.isnan(t)] = np.nanmean(p), np.nanmean(t)
        pt_dict[month]["PRECIP"], pt_dict[month][target_col] = p, t
    
    # autocorrelation/autoregression of the precipitation and temperature data
    pt_dict = investigate_autocorrelation(pt_dict, ar_lag) 
   
    # stationarity of the precipitation and temperature residuals
    # -- note: Tootoonchi (2022) suggests that for exploratory applications
    # -- that stationarity is not necessarily something to correct for, and
    # -- can be considered a part of the conditional variability between the
    # -- two marginals. We take that approach below, so this function only
    # -- identifies/validates the existing stationarity of the residuals,
    # -- if there is any
    if do_validation and ("copula" in validation_figures):
        validate_pt_stationarity(validation_dirpath, validation_extension, pt_dict, target_col)

    # dependence structure of the residuals to help determining copula families
    if do_validation and ("copula" in validation_figures):
        validate_pt_dependence_structure(validation_dirpath, validation_extension, pt_dict, target_col)

    # fit the copula families
    copulaetemp_fit_dict = fit_copula_families(pt_dict, copula_families)

    return copulaetemp_fit_dict

