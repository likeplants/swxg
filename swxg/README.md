# `swxg`
`swxg` is a Python package for modern [stochastic weather generation](https://www.ipcc-data.org/guidelines/pages/weather_generators.html). It is suitable for any use case where traces of precipitation, temperature, and its internal variability across a single or multiple sites impact the model outcomes to be investigated. It expands on existing generators which are often designed for more niche applications like replicating weather regimes, downscaling global circulation models, or using weather as an intermediate step in determining flood or drought indices.

All that is needed to use `swxg` is a set of data with precipitation and temperature observations, one or more locations where the observations were collected, and a timestamp for each of the collected observations. `swxg` quickly generates arbitrarily-long sequences of monthly or daily weather variables that match the spatial and temporal correlations from input observations by: 
  1. fitting observed precipitation individually to a (Gaussian mixture) hidden Markov model with 1 or more hidden states; 
  2. fitting both observed precipitation and temperature with hydroclimatic copulas;
  3. sampling precipitation from its fit, disaggregating to finer resolution where necessary, and;
  4. conditionally sampling temperature from the sampled precipitation and its fit, disaggregating to finer resolution where necessary

## Dependencies
The required dependencies to use `swxg` are:
  * `Python >= 3.10`
  * `copulae >= 0.8`
  * `copulas >= 0.10, < 0.12`
  * `hmmlearn >= 0.3`
  * `matplotlib >= 3.8`
  * `numpy == 2.0`
  * `pandas >= 2.1`
  * `scikit-learn >= 1.4`
  * `scipy >= 1.15`
  * `statsmodels >= 0.14, < 0.15`

Note that these required packages will be automatically downloaded when you install this package.

## Installation
To install `swxg` from PyPI with `pip`:

    pip install swxg

Alternatively, you can install from this repository:

    git clone https://github.com/xthames/swxg.git
    cd swxg
    pip install .

## Contributing, Reporting Issues, and Seeking Support
To **contribute**, please fork this repository and create your own branch. If you are unfamiliar with that process [the corresponding documentation on how to do so from FirstContributions](https://github.com/firstcontributions/first-contributions#first-contributions) is a good place to start. To **report issues** and **seek support**, please use the [GitHub Issues](https://github.com/xthames/swxg/issues) tab for this repository.

## Important Links
  * [Official Source Code](https://github.com/xthames/swxg)
  * [Documentation](https://swxg.readthedocs.org)

If your work uses `swxg`, please cite: 
  * [JORS PAPER IN REVIEW], specifically for the software
  * [Thames et al. (2026, preprint)](https://doi.org/10.22541/essoar.176840902.22785610/v1), if relevant to applied (first) use case

## Known Model Limitations
Because `swxg` is a semi-parametric model, the quality of the input dataset will be reflected in: (1) the confidence of the fits for precipitation and the copulas, and; (2) the resolution of the generated weather. 
  1. To fit precipitation and the copulas `swxg` aggregates precipitation and temperature both annually and monthly, meaning that more complete years of input data will produce better fitting. A `UserWarning` will appear **if you use fewer than 30 years of input data**; another `UserWarning` will appear **if more than 20 sites are considered**. Fitting will still procede regardless, but it is strongly recommended to validate the precipitation and copula fitness through additional metrics if either condition is met.
  2. When generating weather `swxg` gives the option to determine its output resolution, either at the `monthly` or `daily` scale. How resolved the generated weather can be is determined by the input dataset: `monthly` inputs can be resolved to `monthly` outputs; `daily` inputs can be resolved to both `daily` and `monthly` outputs. A `UserWarning` will appear when **trying to resolve `daily` outputs from `monthly` inputs**. If attempted, the `monthly` resolution will output instead. Subdaily inputs are accepted but generating at the subdaily scale is not yet implemented, and so subdaily data is aggregated to daily.  
