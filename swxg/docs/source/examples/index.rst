Tutorial and Examples
=====================

The following material provides a step-by-step instruction set to understand how ``swxg`` works using the built-in test datasets. We will use daily observations for this tutorial, but the process is the same for both test datasets. The summary for this tutorial follows:

.. code-block:: python

    import swxg
    model = swxg.SWXGModel(swxg.test_wx.daily)
    model.fit()
    wx = model.synthesize()

Importing ``swxg``
------------------

Importing everything from the ``swxg`` package is easy:

.. code-block:: python

    import swxg

The ``swxg`` package only has two user-facing objects: 

 * ``swxg.SWXGModel``: a class that fits your observations, validates the fit, generates new data from those observations and fits, and compares the generated data to the input data.
 * ``swxg.test_wx``: the object that holds the test datasets that we'll work with for the tutorial. If you are familiar with using ``swxg`` you do not need to import this.

The ``test_wx.daily`` DataFrame
-------------------------------

The ``swxg.test_wx.daily`` dataset is a Pandas dataframe that when using ``print(swxg.test_wx.daily)`` should look like this:

=====  ====  ==========  ========  =========
 ..    SITE   DATETIME    PRECIP     TEMP
=====  ====  ==========  ========  =========
  0     X    1968-01-01   0.0127   -3.330000
  1     X    1968-01-02   0.0000   -10.00000
  2     X    1968-01-03   0.0000   -3.330000
  3     X    1968-01-04   0.0000    1.110000
  4     X    1968-01-05   0.0000   -6.670000
 ...   ...       ...       ...        ...
41471   Y    2025-10-08   0.0005   15.233600
41472   Y    2025-10-09   0.0064   15.138958
41473   Y    2025-10-10   0.0000   14.945686
41474   Y    2025-10-11   0.0000   14.388200
41475   Y    2025-10-12   0.0010   14.547647
=====  ====  ==========  ========  =========

.. |deg| unicode:: U+00B0
 
Let's parse this dataframe:

 * There are four columns: ``SITE``, ``DATETIME``, ``PRECIP``, and ``TEMP``. The ``swxg.SWXGModel`` class expects at least four columns with these names specifically, otherwise it won't know how to format, process, fit, or generate data. **It also requires this order for the columns as well**. The current version of the generator will only generate precipitation and temperature, but in the future it may be able to do more.  
 * The ``SITE`` column has type ``str`` and has a unique identifier for each unique site. Letters ``X`` through ``Y`` are used here, but full strings can also be used.
 * The ``DATETIME`` column has type ``datetime``. This is the standard object output using ``datetime.datetime.strptime(date_string, input_format_code)`` from the ``datetime`` package. You must format the date using YYYY-MM-DD (so, hyphens, not forward slashes). `See the datetime documentation for more information <https://docs.python.org/3/library/datetime.html#format-codes>`__.
 * The ``PRECIP`` column has type ``float``, and is reported in units of [m]. Units must be in metric.
 * The ``TEMP`` column has type ``float``, and is reported in units of [\ |deg|\ C]. Units must be in metric.

And that's it -- just a location, timestamp and linked precipitation and temperature are the only datapoints you need to get started (even with your own data)! If one of the columns has the wrong name, type, or the column is in the wrong location, the Python editor will throw an error until it is corrected and acceptable for the generator.

Applying an Input DataFrame to ``SWXGModel``
--------------------------------------------

Taking an input dataframe and priming the generator with it is trivial:

.. code-block:: python

    model = swxg.SWXGModel(swxg.test_wx.daily)

This creates an instance of the ``SWXGModel`` class with ``test_wx.daily`` as the initial input. **You cannot instantiate the class without an input dataframe**.

The ``SWXGModel`` class has the following (if initially empty) attributes:

 * ``raw_data``: This is the dataframe you gave it as input, exactly as it was input. This is here as a sanity check that your input successfully made it into the model without artifacts.
 * ``data``: This is the input dataframe, reformatted to separate the ``DATETIME`` column into ``YEARS``, ``MONTHS``, and potentially ``DAYS``, depending on the input dataframe resolution. Starting in version 0.2.2 you can include subdaily data but it will be aggregated to daily (a subdaily generation scheme has yet to be implemented).
 * ``resolution``: This is the resolution of the input dataframe as determined by the model. It can be ``daily`` or ``monthly``.
 * ``precip_fit_dict``: This is the dictionary of statistics related to fitting precipitation that the generator will use. It is initially ``{}``.
 * ``copulaetemp_fit_dict``: This is the dictionary of statistics related to fitting copulae and temperature that the generator will use. It is initially ``{}``.
 * ``is_fit``: This is a flag for whether or not the input data has been fit yet. **Generation cannot happen without the data having been previously fit**. It is initially ``False``.

Displaying ``model.data`` using ``print(model.data)`` should look like this:

=====  ====  ====  =====  ===  ========  =========
 ..    SITE  YEAR  MONTH  DAY   PRECIP     TEMP
=====  ====  ====  =====  ===  ========  =========
  0     X    1968     1    1    0.0127   -3.330000
  1     X    1968     1    2    0.0000   -10.00000
  2     X    1968     1    3    0.0000   -3.330000
  3     X    1968     1    4    0.0000    1.110000
  4     X    1968     1    5    0.0000   -6.670000
 ...   ...   ...    ...   ...     ...       ...
41471   Y    2025    10    8    0.0005   15.233600
41472   Y    2025    10    9    0.0064   15.138958
41473   Y    2025    10    10   0.0000   14.945686
41474   Y    2025    10    11   0.0000   14.388200
41475   Y    2025    10    12   0.0010   14.547647
=====  ====  ====  =====  ===  ========  =========

with ``model.resolution == 'monthly'``. The determination of the ``monthly`` or ``daily`` resolution comes from the set of day values in the original ``DATETIME`` raw data column. If you are using monthly data but have multiple different numbered days in that column, the generator will assume you are inputting daily data. Picking a single day for all data---it doesn't matter which---will assume monthly data.

.. danger::

    It is permissible to overwrite the model attributes, if you are comfortable with doing so and understand how fitting and/or generation works. **It is recommended that you do not** and let the generator do this for you.

Fitting Data
------------

Fitting the reformatted input data is as easy as:

.. code-block:: python

    model.fit()

Using the :meth:`fit() <swxg.SWXGModel.fit>` method will first fit the preciptation data and then the copula/temperature data. It returns nothing and only updates the internal attributes. You can confirm that both precipitation and copulas/temperature have been fit by (1) checking that ``model.is_fit == True`` and (2) observing the output to screen. The output to screen is a clean version of ``model.precip_fit_dict`` and ``model.copulaetemp_fit_dict`` and should look similar the following:

.. code-block:: text

    Positive definite covariance matrix for GMHMM fit found for 1 state(s)!
    Positive definite covariance matrix for GMHMM fit found for 2 state(s)!
    Positive definite covariance matrix for GMHMM fit found for 3 state(s)!
    Positive definite covariance matrix for GMHMM fit cannot be found for 4 states...
    --------------- Precipitation Fit ---------------
    * Number of GMHMM States: 1

    * GMHMM Means/Stds/Goodness of Fit per Site and State
    STATE SITE     MEANS     STDS    (AD, CvM, KS) P-Value
        0    X -0.050047 0.117816 (0.4566, 0.3605, 0.1867)
        0    Y  0.044184 0.108240 (0.5203, 0.5926, 0.4272)

    * Transition Probability Matrix
                 TO STATE 0
    FROM STATE 0        1.0
    -------------------------------------------------

    ------------------ Copulas Fit ------------------
    Copula Statistics for: JAN
    * Best-Fitting Copula Family: Frank
    * All Family Parameters and Fit Comparison
                  Hyperparameter       AIC Cramér von Mises Kolmogorov-Smirnov (CvM, KS) P-Value
    Independence             NaN  0.000000         0.076976           0.071427      (0.1, 0.235)
    Frank               1.354955 -0.690616         0.026624           0.055779    (0.505, 0.375)
    Gaussian            0.219533 -1.549539         0.032455           0.059263     (0.545, 0.44)
    
    Copula Statistics for: FEB
    ...
    -------------------------------------------------

.. |eacute| unicode:: U+00E9

The critical fitness parameters for precipitation are how many states were chosen by the GMHMM, the means and standard deviations of the GMHMM per site and state, and the transition probability matrix. These are fairly easy to interpret, though note that the precipitation data behind the scenes has been log\ :sub:`10`\ -transformed and so the means can be negative and standard deviations reflect this transformation. The column that includes the p-values indicates whether or not a Gaussian distribution with the fitted mean and standard deviation is statistically indistinguishable from the observed data using three different tests (`Anderson-Darling (AD), Cramer von Mises (CvM), Kolmogorov-Smirnov (KS) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html#r48df2cf935d3-1>`__). Values above 0.05 indicate that these distributions are statistically indistinguishable. Note that these tests are bootstrapped, which means the values shown here may differ slightly between those on your readout.

The critical fitness parameters for the copulas are which month is being fit and the best fitting copula family using three different metrics (AIC, Cram\ |eacute|\ r von Mises, and Kolmogorov-Smirnov). Smaller numbers for all three metrics indicate better fitness, and any AIC value within 2 of another should be considered an equivalent fitness. In this case for January the Frank copula is the smallest across two of the metrics and therefore it is determined to be the best choice, although Frank and Gaussian perform similarly. Note that, again, the Cram\ |eacute|\ r von Mises and Kolmogorov-Smirnov metrics and p-values are bootstrapped and so there may be small differences between the values listed here and those on your readout.

.. note::

    ``swxg.test_wx.daily`` may occasionally find a valid fit with 4 states. This is because the GMHMM state fitting algorithm checks a large-but-finite number of models with random initializations before moving on to the next number of states. The seed for each search is set via `RNG seed <https://numpy.org/doc/2.2/reference/random/generator.html#numpy.random.Generator>`__, so you can guarantee the same best fitting number of states by setting this seed before fitting the data. **The fitting and generating procedure is the same regardless of how many states are found**.

Using the default of no arguments to :meth:`fit() <swxg.SWXGModel.fit>` produces 12 validation figures, 3 for the fit regarding precipitation and 9 for the fit regarding the copulas. Each can help make a more-informed determination about how the fitting was done and if a better fit is possible (see :ref:`How to Interpret the Validation Figures <how-to-validate>` for more information). This can be accomplished by interfacing with the arguments and keyword arguments accepted by the :meth:`fit() <swxg.SWXGModel.fit>` method. These include, but are not limited to, turning off the output statistics display (``verbose=False``), turning off the validation figures (``validate=False``), and hard-setting the number of GMHMM states to use and restricting the copula families to try (e.g., ``kwargs={"gmhmm_states": 1, "copula_families: ["Frank"]}``). Please review the method to learn the default behavior and how to change it, though for this Tutorial we will leave it unchanged.


Generating (Synthesizing) Data
------------------------------

Generating data from the fit is just as easy as fitting the data:

.. code-block:: python

    wx = model.synthesize()

Using the :meth:`synthesize() <swxg.SWXGModel.synthesize>` method returns a dataframe of precipitation and temperature generated from the fit statistics. The default behavior is also to validate the generated weather against the observed weather, and the number of produced plots to do this varies based on the number of sites in the dataset; this can take a while, especially for many years of daily data. This method also takes several additional arguments which should be reviewed (but again are outside the scope of this Tutorial).

``print(wx)`` will have the general form:

=====  ====  ====  =====  ===  ===============  ===============
 ..    SITE  YEAR  MONTH  DAY      PRECIP             TEMP
=====  ====  ====  =====  ===  ===============  ===============
  0     X      1     1     1     p\ :sub:`1`      T\ :sub:`1`
  1     X      1     1     2     p\ :sub:`2`      T\ :sub:`2`
  2     X      1     1     3     p\ :sub:`3`      T\ :sub:`3`
  3     X      1     1     4     p\ :sub:`4`      T\ :sub:`4`
  4     X      1     1     5     p\ :sub:`5`      T\ :sub:`5`
...    ...   ...    ...   ...        ...             ...
41605   Y     102   12     27  p\ :sub:`41605`  T\ :sub:`41605` 
41606   Y     102   12     28  p\ :sub:`41606`  T\ :sub:`41606` 
41607   Y     102   12     29  p\ :sub:`41607`  T\ :sub:`41607` 
41608   Y     102   12     30  p\ :sub:`41608`  T\ :sub:`41608` 
41609   Y     102   12     31  p\ :sub:`41609`  T\ :sub:`41609`
=====  ====  ====  =====  ===  ===============  ===============

This has the same format as the reformatted input dataframe, with some key differences: 

 * The ``YEAR`` column has been replaced with a value representing the year in order of the sequence it was generated. This is because the generated data reflect the statistics from the entire observation set and therefore could align to any observed year.
 * The size of the dataframe increased. This is because generated data does not contain NaNs or empty rows, where the input dataset might. The generator will default to generating the number of years given to it in the input set unless otherwise specified by the ``n`` argument.
 * You can synthesize weather at as-fine or coarser resolutions than your input dataset using the ``resolution`` argument, but not finer. Attempting finer resolutions will default to the resolution of the input dataset.
 * The ``PRECIP`` and ``TEMP`` columns will be unique for each random seed. Again, fixing the RNG seed can guarantee reproducibility.

.. note::

    While the generator was designed to fit and synthesize weather variables across multiple sites, it will still function without issue for just a single site. That said, with only one site certain validation and comparison figures that look at metrics like the correlations between sites will produce trivial results (i.e., the spatial correlation between site A and site A for precipitation is 100%). 

Next Steps
----------

And that's all there is to it! You can try generating a new sample simply by envoking ``wx2 = model.synthesize()``, or :ref:`use a pretreated dataset of your own <pretreat>`. We also recommend looking at :ref:`How to Interpret the Validation Figures <how-to-validate>` and the :ref:`API <api>` next in order to get the best possible fits.
