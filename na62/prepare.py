import numpy as np
import pandas as pd
import uproot

from .hlf import compute_eop



def compute_derived(df: pd.DataFrame) -> None:
    compute_eop(df, 1)
    compute_eop(df, 2)
    compute_eop(df, 3)
