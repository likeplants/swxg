.. _how-to-validate:

How to Interpret the Validation Figures
=======================================

When you eventually want to look at the validation figures, the sheer number can be overwhelming. This page will help you make sense of each of these, and from which function they derive.

.. tip::

   In both the :meth:`fit() <swxg.SWXGModel.fit>` and :meth:`synthesize() <swxg.SWXGModel.synthesize>` methods there is an argument called ``dirpath``. You can give it a string *relative to the current working directory* like ``"path/to/validation/"`` and that is where all of the figures created by that method will go. Setting the path in one method will not set it in the other; this is to give the user more control over where these figures end up.

.. tip::

   All of the validation figures have ``.svg`` extensions by default. This is so that you can zoom in on the figures and pick things apart by eye, or remove elements for clarity in the appropriate editor. You may change the extensions using the ``figure_extension`` keyword argument.

.. tip::

  You may not want to validate all aspects of the fitting process, especially after you have found a combination of arguments that provides a suitable fit. The default behavior is to create figures for everything, but you may specify whether you want only precipitation or copula figures in the ``validation_figures`` keyword argument.  

:func:`validate_gmhmm_states() <swxg.make_figures.validate_gmhmm_states>`
---------------------------------------------------------------------------

 * **How to Interpret**: A test of how good the GMHMM fit is for each of the explored number of states. Note that the loglikelihood statistic is monotonically increasing and the AIC statistic can run into issues with overfitting, so the BIC is generally the best choice. The lowest value corresponds to the best fit. If you set the number of states and don't explore any this will not be called.
 * **Validates**: Precipitation
 * **Output**: ``Validate_GMHMM_NumStates.svg``

:func:`validate_explore_pt_dependence() <swxg.make_figures.validate_explore_pt_dependence>`
-------------------------------------------------------------------------------------------

 * **How to Interpret**: A plot of the Kendall and Spearman correlations between precipitation and temperature, and a scatterplot of precipitation against temperature. Both precipitation and temperature have been spatially averaged, so axes ticks or panels in each figure are per month. The metrics should align with the relative position of the Kendall Plots against the 1:1 line.
 * **Validates**: Copulas
 * **Output**: ``Validate_Copulae_ExplorePTCorrelation_MonthlySpatialAverage.svg``, ``Validate_Copulae_PTDistribution_MonthlySpatialAverage.svg``

:func:`validate_pt_acf() <swxg.make_figures.validate_pt_acf>`
-------------------------------------------------------------

 * **How to Interpret**: Autocorrelation functions of spatially-averaged precipitation and temperature, and their residuals. The residuals should not have any significant autocorrelations (points are within the bands).
 * **Validates**: Copulas
 * **Output**: ``Validate_Copulae_[Precip,Temp]_ACF.svg``

:func:`validate_pt_stationarity() <swxg.make_figures.validate_pt_stationarity>`
-------------------------------------------------------------------------------

 * **How to Interpret**: A test of the stationarity of the residuals using both the `Augmented Dickey-Fuller (ADF) test <https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html>`__ and the `Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test <https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html>`__. P-Values above 0.05 imply that the residuals are stationary. Note that the ADF test is transformed because its null hypothesis (that the data are not stationary) is the opposite of the KPSS test (that the data are stationary).
 * **Validates**: Copulas
 * **Output**: ``Validate_Copulae_[Precip,Temp]_ResidStationarity.svg``

:func:`validate_pt_dependence_structure() <swxg.make_figures.validate_pt_dependence_structure>`
-----------------------------------------------------------------------------------------------

 * **How to Interpret**: A test of the dependence structure of the precipitation and temperature residuals to determine which copula families can be used. These should rhyme with the Kendall statistics; you can read more about them here: `Genest & Boies (2003) <https://www.jstor.org/stable/30037296>`__.
 * **Validates**: Copulas
 * **Output**: ``Validate_Copulae_KPlots.svg`` 

:func:`validate_gmhmm_statistics() <swxg.make_figures.validate_gmhmm_statistics>`
-----------------------------------------------------------------------------------

 * **How to Interpret**: Various statistics related to the fitting of the precipitation GMHMM. Q-Q plots show how Gaussian the log\ :sub:`10`\ -transformed precipitation data is; ACFs/PACFs show if the hidden states are Markovian (only plots if the number of determined hidden states is greater than 1); the transition probability matrix shows the likelihood of transition between hidden states.
 * **Validates**: Precipitation
 * **Output**: ``Validate_GMHMM_QQs.svg``, ``Validate_GMHMM_HiddenStateMarkovStructure.svg``, ``Validate_GMHMM_TransitionProbabilities.svg``

:func:`validate_copulae_statistics() <swxg.make_figures.validate_copulae_statistics>`
-------------------------------------------------------------------------------------

 * **How to Interpret**: Various statistics related to the fitting of the copulae. The best-fitting copula families per month are shown in the radial plot, with lowest values representing the best fit. In the contour plot, the various copula families (colors) are compared to the empirical copula (black).
 * **Validates**: Copulas
 * **Output**: ``Validate_Copulae_FitMetrics.svg``, ``Validate_Copulae_Comparison.svg``

:func:`compare_synth_to_obs() <swxg.make_figures.compare_synth_to_obs>`
-----------------------------------------------------------------------

 * **How to Interpret**: A comparison of all the generated data against the observed data. Observed data is in black and generated data is in grey. A successfully fit SWG will have the following comparisons between generated weather variables: generated histograms should be largely contained within observed histograms but extend slightly farther off to both sides; scatterplots and cumulative frequencies of generated data should envelop the observed data, and; correlation and statistical metrics should either approximately match observations or have p-values greater than 0.05.
 * **Validates**: Generated weather to observed weather
 * **Output**: ``Compare_AnnualPrecip.svg``, ``Compare_CumulativeFrequency_Precip.svg``, ``Compare_SpatialCorrelations_[MONTH].svg``, ``Compare_TemporalCorrelations_[SITE].svg``, ``Compare_PTCorrelations_KendallSpearman.svg``, ``Compare_HistScatter_[SITE].svg``, ``Compare_StatisticalDistributions_[SITE].svg``, ``Compare_PerDOY_[SITE].svg``, ``Compare_StatisticalConvergence_[SITE].svg``, ``Compare_ExtremeValues_[SITE].svg``

