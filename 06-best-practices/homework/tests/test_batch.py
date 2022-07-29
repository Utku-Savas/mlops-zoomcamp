from datetime import datetime
import pandas as pd

import pytest
from deepdiff import DeepDiff

import batch

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    actual_df = batch.prepare_data(df, columns)
    
    expected_data = [
        ("-1", "-1", dt(1, 2), dt(1, 10), 8.0),
        ("1", "1", dt(1, 2), dt(1, 10), 8.0),    
    ]

    expected_columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime', 'duration']
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    assert {} == DeepDiff(actual_df, expected_df, ignore_order=True) 
    