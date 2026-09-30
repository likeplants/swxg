``swxg``
==========================

.. note::

    This project is under active development.

``swxg`` is a Python package for modern `stochastic weather generation <https://www.ipcc-data.org/guidelines/pages/weather_generators.html>`__. It is suitable for any use case where traces of precipitation, temperature, and its internal variability across a single or multiple sites impact the model outcomes to be investigated. It expands on existing generators which are often designed for more niche applications like replicating weather regimes, downscaling global circulation models, or using weather as an intermediate step in determining flood or drought indices.

All that is needed to use ``swxg`` is a set of data with precipitation and temperature observations, one or more locations where the observations were collected, and a timestamp for each of the collected observations. ``swxg`` quickly generates arbitrarily-long sequences of monthly or daily weather variables that match the spatial and temporal correlations from input observations using hidden Markov models, hydroclimatic copulas, and *k*-NN disaggregation techniques. 


.. toctree::
    :titlesonly:
    :maxdepth: 1
    :caption: Table of Contents

    Getting Started <getting-started>
    Tutorial/Examples <examples/index>
    Pretreating Your Own Dataset <pretreatment/index>
    How to Interpret Validation Figures <validation/index>
    I/O Control <iocontrol/index>
    API <api/index>

.. admonition:: Cite the paper!

    [JORS IN REVIEW]

