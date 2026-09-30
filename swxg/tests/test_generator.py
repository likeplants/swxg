import numpy as np
import pandas as pd
import datetime as dt

import pytest
from swxg.model import SWXGModel
from swxg.test_wx import monthly, daily


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator() -> None:  
    #model = SWXGModel(monthly) 
    model = SWXGModel(daily) 
    #model.fit()
    model.fit(validate=False, kwargs={"gmhmm_states": 1, "copula_families": ["Frank"]})
    synth_wx = model.synthesize(kwargs={"validation_samplesize_mult": 15})
