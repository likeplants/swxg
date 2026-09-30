.. _iocontrol:

I/O Control
===========

It is possible to save and load both the fitted model and the generated sequence(s) of precipitation and temperature, although this functionality has not been explicitly included in ``swxg`` to give greater control to the user over how these objects integrate with their own workflows. The following provides some examples of how the the model and the generated data could be saved and loaded.

Let's assume we have two objects we're trying to save:

 * ``model``, which is the fitted model
 * ``wx``, which is the dataframe that contains all of the generated sequences of precipitation and temperature across all sites by ``model``

Saving
------
Saving ``model`` is simple with the ``pickle`` module (which is a standard Python library):

.. code-block:: python

    # import pickle
    import pickle
     
    # save `model` using pickle 
    filename = "/path/to/dir/swxg_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)

Saving ``wx`` is also simple since it is a Pandas dataframe, which means all methods that work to save dataframes work here as well:

.. code-block:: python

    filename = "/path/to/dir/wx"    

    # save `wx` as .csv (this will not save information about the columns like typing)
    extension = ".csv"
    wx.to_csv(filename + extension, index=False)
    
    # save `wx` as .pkl (import pickle to do this)
    import pickle
    extension = ".pkl"
    wx.to_pickle(filename + extension)

    # save `wx` as `.parquet` (import a Parquet library like pyarrow to do this)
    import pyarrow
    extension = ".parquet"
    wx.to_parquet(filename + extension)

Loading
-------
Loading the saved ``model`` can also be done with ``pickle``:

.. code-block:: python
    
    with open("/path/to/dir/swxg_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

Both ``model`` and ``loaded_model`` will be identical instances of ``SWXGModel``, and thus ``loaded_model`` can be used to generate new sequences of precipitation and temperature based on the inputs given to ``model``.

Loading the generated dataframe should also be done through the native Pandas methods:

.. code-block:: python

    # need to recover column type information
    wx_csv = pd.read_csv("/path/to/dir/wx.csv")
    wx_csv = wx_csv.astype({"SITE": str, "YEAR": int, "MONTH": int, "PRECIP": float, "TEMP": float})
    
    # no need to recover column type information, already saved
    wx_pkl = pd.read_pickle("wx.pkl")

    # no need to recover column type information, already saved
    wx_parquet = pd.read_parquet("wx.parquet")

By reading this data back in with the native Pandas methods, all three saved generated sequences should be identical to the original ``wx`` dataframe.
