.. _pretreat:

Pretreating Your Own Dataset
============================

``swxg`` is meant to accept any set of observations of precipitation and temperature, not just the test sets included with the library. However, in order to use ``swxg`` for this purpose, it is extremely likely that an input dataset will need to be cleaned (or "pretreated") first. This section will help you understand why that's necessary and how to do it.

Guiding Principles of Pretreatment
----------------------------------

.. |deg| unicode:: U+00B0

There are many ways to pretreat data, and no method is intrinsically better than another. That said, there are three key ideas that you should keep in mind when pretreating data for ``swxg``:

 * The final dataset is a singular dataframe with (at least) four columns titled ``SITE``, ``DATETIME``, ``PRECIP``, ``TEMP``, in that order. The datatypes for those columns should be ``str`` or ``object``, ``datetime64[ns]``, ``float``, and ``float``, respectively.
 * Precipitation and temperature columns should be in units of [m] and [\ |deg|\ C], respectively.
 * Missing (precipitation, temperature, or both are NaNs) or incomplete (sites do not coverage all the same dates) entries should either be infilled from existing data/secondary sites or outright removed. This is necessary because the generation scheme uses a non-parametric (*k*-NN) disaggregation algorithm and therefore leaving missing or incomplete data in the observations can potentially populate the generated data with these artifacts; fitting is unaffected. Trying to produce validation figures for missing or incomplete data is not possible and may cause an error. 

Basic Pretreatment Procedure
----------------------------

To begin, let's assume that we have a single file called ``weather.csv`` that contains all of the data for all of the sites. That file can have any number of columns, but the critical columns are:

 * **the site name**, which acts as the spatial distinguishing feature. The column in the file is labeled something like ``NAME``
 * **the datetime**, which acts as the temporal stamp on when the data were collected. The column in the file is labeled something like ``DATE``
 * **precipitation**, which is self-explanatory. The column in the file is labeled something like ``PRCP``
 * **temperature**, which might come as an average (something like ``TAVG``) or as a pair of minimums and maximums (something like ``TMIN`` and ``TMAX``)  

If you have multiple columns about temperature but no average, the first step is to create a column of average values at the same timestep as the rest of the data:

.. code-block:: python

    # import appropriate libraries
    import numpy as np
    import pandas as pd    
    import datetime as dt

    # load in weather.csv file as a dataframe
    raw_df = pd.read_csv("weather.csv")

    # create a column called "TAVG", which is the average of the "TMIN" and "TMAX" columns
    raw_df["TAVG"] = raw_df[["TMIN", "TMAX"]].mean(axis=1)

If the data has imperial units, these must be converted to metric:

.. code-block:: python

    # convert from imperial [inches, degF] to metric [meters, degC]
    # -- remove TMIN and TMAX if they are not in your dataframe
    for col in ["PRCP", "TAVG", "TMIN", "TMAX"]:
        if col == "PRCP":
            raw_df[col] = raw_df[col].values * 0.0254 
        else:
            raw_df[col] = (raw_df[col].values - 32.) * (5./9)

The ``DATE`` column must also be converted to a specific format and datatype:

.. code-block:: python

    # formatting the DATE information
    # -- change the format code in strptime() to match yours
    # -- you can find more about the format codes here: https://docs.python.org/3/library/datetime.html#format-codes
    dates = []
    for i in range(raw_df.shape[0]):
        row_entry = raw_df.iloc[i]
        row_date = row_entry["DATE"]
        date_raw = dt.datetime.strptime(row_date, "%YOUR/%FORMAT/%HERE")
        if date_raw.year > dt.datetime.now().year:
            date_raw = dt.datetime(date_raw.year - 100, date_raw.month, date_raw.day)
        dates.append(date_raw.strftime("%Y-%m-%d"))
    raw_df["DATE"] = pd.to_datetime(dates)

The columns in the dataframe should be renamed like so:

.. code-block:: python

    # replace with the actual column headers in your dataframe as appropriate
    raw_df.rename(columns={"NAME": "SITE", "DATE": "DATETIME", "PRCP": "PRECIP", "TAVG": "TEMP"}, inplace=True)

Your dataset may come with timestamps that contain missing observations or observations which exist for one site that do not exist for others. Incomplete years with missing months or days may ultimately be a problem in the generation step, which expects full years to perform the non-parametric resampling. Thus, missing values should be handled during pretreatment. It's best to first infill missing data from the existing dataset using averages of existing months or days, if possible. You can check how many of your values are missing using:

.. code-block:: python

    # compare total entries to non-null count
    raw_df.info()

If the difference between the ``Non-Null Count`` and the total number of ``Entries`` is small (read: please use your own best judgment here on what constitutes "small"), you can use the following algorithms to infill the missing data. For a dataset at the ``monthly`` resolution the code to do this looks like the following:

.. code-block:: python

    avg_dict = {}
    sites = sorted(set(raw_df["SITE"].values))

    # structure each monthly datapoint into a dictionary by site, month
    months = [int(pd.to_datetime(raw_df.iloc[i]["DATETIME"]).month) for i in range(raw_df.shape[0])]
    for site in sites:
        if site not in avg_dict: avg_dict[site] = {}
        site_idx = raw_df["SITE"] == site 
        site_entry = raw_df.loc[site_idx]
        for month in months:
            month_idx = [int(pd.to_datetime(raw_df.iloc[i]["DATETIME"]).month) == month for i in range(site_entry.shape[0])]
            avg_dict[site][month] = {"precip": [raw_df.loc[site_idx & month_idx, "PRECIP"].values],
                                     "temp": [raw_df.loc[site_idx & month_idx, "TEMP"].values]}
    
    # per row, fill NaNs with monthly averages
    for i in range(raw_df.shape[0]):
        row_entry = raw_df.iloc[i]
        site, month = row_entry["SITE"], int(pd.to_datetime(row_entry["DATETIME"]).month)
        if np.isnan(row_entry["PRECIP"]): raw_df.at[i, "PRECIP"] = float(np.nanmean(avg_dict[site][month]["precip"]))
        if np.isnan(row_entry["TEMP"]): raw_df.at[i, "TEMP"] = float(np.nanmean(avg_dict[site][month]["temp"]))
     
For datasets at the ``daily`` resolution, the equivalent process is:

.. code-block:: python

    avg_dict = {}
    sites = sorted(set(raw_df["SITE"].values))

    # structure each daily datapoint into a dictionary by site, doy
    doys = [int(pd.to_datetime(raw_df.iloc[i]["DATETIME"]).dayofyear) for i in range(raw_df.shape[0])] 
    for site in sites:
        if site not in avg_dict: avg_dict[site] = {}
        site_idx = raw_df["SITE"] == site
        site_entry = raw_df.loc[site_idx]
        for i in range(site_entry.shape[0]):
            row_entry = site_entry.iloc[i]
            doy = int(pd.to_datetime(row_entry["DATETIME"]).dayofyear)
            if doy not in avg_dict[site]:
                avg_dict[site][doy] = {"precip": [row_entry["PRECIP"]], "temp": [row_entry["TEMP"]]}
            else:
                avg_dict[site][doy]["precip"].append(row_entry["PRECIP"])
                avg_dict[site][doy]["temp"].append(row_entry["TEMP"])
    for site in avg_dict:
        for doy in avg_dict[site]:
            avg_dict[site][doy]["precip"] = np.nanmean(avg_dict[site][doy]["precip"])
            avg_dict[site][doy]["temp"] = np.nanmean(avg_dict[site][doy]["temp"])
    
    # per row, fill NaNs with doy averages
    for i in range(raw_df.shape[0]):
        row_entry = raw_df.iloc[i]
        site, doy = row_entry["SITE"], int(pd.to_datetime(row_entry["DATETIME"]).dayofyear)
        if np.isnan(row_entry["PRECIP"]): raw_df.at[i, "PRECIP"] = float(avg_dict[site][doy]["precip"])
        if np.isnan(row_entry["TEMP"]): raw_df.at[i, "TEMP"] = float(avg_dict[site][doy]["temp"])
     
If too much (there is no hard rule for this, but maybe something like more than half) of the dataset is missing or you cannot infill data from the existing/external sources, you can simply remove the offending entries. **Please be careful when bulk removing data as this may dramatically reduce the fitness of the model; referring to the validation figures is imperative when removing data like this**. The code to do this looks like the following:

.. code-block:: python

    # remove missing data
    raw_df.dropna(axis=0, inplace=True)

Dates with only some sites reporting or years with only some recorded months should now be removed: 
 
.. code-block:: python
    
    # remove periods when only some sites have data
    good_dates = []
    for date in sorted(set(raw_df["DATETIME"].values)):
        date_idx = raw_df["DATETIME"] == date
        date_entry = raw_df.loc[date_idx]
        if date_entry.shape[0] == len(sites):
            good_dates.append(date)
    clean_df = raw_df[raw_df["DATETIME"].isin(good_dates)]
    clean_df.reset_index(drop=True, inplace=True)

    # remove years with fewer than 12 months
    good_year_dict, indices_to_remove = {}, []
    for date in sorted(set(clean_df["DATETIME"].values)):
        dtstamp = pd.to_datetime(date)
        year, month = int(dtstamp.year), int(dtstamp.month)
        if year not in good_year_dict:
            good_year_dict[year] = [month]
        else:
            good_year_dict[year].append(month)
    for year in good_year_dict:
        good_year_dict[year] = list(set(good_year_dict[year]))
    for i in range(clean_df.shape[0]):
        row_entry = clean_df.iloc[i]
        year = int(row_entry["DATETIME"].year)
        if len(good_year_dict[year]) < 12:
            indices_to_remove.append(i)
    clean_df.drop(index=indices_to_remove, inplace=True)
    clean_df.reset_index(drop=True, inplace=True)

Finally, reducing the cleaned dataframe to just the four necessary columns and saving it is simple:

.. code-block:: python

    # drop non-necessary columns
    clean_df = clean_df[["SITE", "DATETIME", "PRECIP", "TEMP"]] 
    
    # you may also want to rename the sites
    clean_df["SITE"] = clean_df["SITE"].map({"VERY LONG SITE NAME #1": "Short1",
                                             "VERY LONG SITE NAME #2": "Short2",
                                             "VERY LONG SITE NAME #3": "Short3"})

    # save the dataframe -- .pkl is recommended because it saves datatypes and is always available in Python environments
    clean_df.to_pickle("clean_wx.pkl")

.. note::

    This is simply one approach to data pretreatment, guided by the input dataframe to ``swxg`` and how both the fitting and generation procedures work. Your dataset may require more pretreatment and cleaning than just the outline provided here.

A Note on Bias-Correction
-------------------------

Additional data sources can also be used to infill missing data, and using external sources can sometimes be preferable to simply removing datapoints. If using "secondary" sites to infill a primary, bias-correction of the secondary site(s) to the primary site(s) of interest is required. Bias-correction algorithms for hydroclimatic variables are a well-studied problem, and you can find some more information on how to do this `for precipitation <https://doi.org/10.1002/joc.2168>`__ and `for temperature <https://doi.org/10.1016/j.heliyon.2024.e40352>`__ at the linked sources. 
