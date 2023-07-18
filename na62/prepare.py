from hlf import compute_eop
import pandas as pd

def compute_derived(df: pd.DataFrame) -> None:
    compute_eop(df, 1)
    compute_eop(df, 2)
    compute_eop(df, 3)