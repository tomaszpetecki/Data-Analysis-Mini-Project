import pandas as pd
import numpy as np
import pandas.api.types
from src.config import TARGET_COLUMN, DATE_COLUMN
from typing import Callable

def fmt(x: float) -> str:
            if abs(x) >= 10:
                return f"{x:.0f}"
            else:
                return f"{x:.1f}"

def bin_data(s:pd.Series,rmin:int=4,rmax:int=10, max_card:int = 10, bins:int|list = 5):
    "bins: categorical Series with nice labels"
    "labels: list of strings (same order as cats)"
    "intervals: IntervalIndex if numeric-binned"

    #If numeric and there is more than 10 unique values ---> bin into n_bins:
        #If skewed data, then use quantile bins
        #Else we will use equal-width bins
    #If not numeric then all leave as is
    
    s = s.copy()
    s = s.dropna()
    is_num = pd.api.types.is_numeric_dtype(s)

    if (not is_num) or s.nunique() <= max_card:
        return s
    
    #Come up with an appropriate range
    if isinstance(bins,int):
        d = bins - 2 #min - rmin --- rmax - max
        main_bins = []
        to_add = (rmax - rmin) % d
        right = rmax
        left = rmin
        for i in range(to_add):
            if i % 2 == 0:
                if right -rmax != -1:
                    right += 1
                else:
                    left -= 1
            else:
                if left -rmin != 1:
                    left -= 1
                else:
                    right += 1
        
        main_bins = list(range(left, right+1,(right-left)//d))
        main_bins = [int(s.min())] + main_bins + [int(s.max())]
    else:
        main_bins = bins

    #make labels
    labels = []
    for i in range(0,len(main_bins)-1):
        if i == 0:
            labels.append(f"<{main_bins[i+1]}")
        elif i == len(main_bins)-2:
            labels.append(f"{main_bins[i]}+")
        else:
            labels.append(f"{main_bins[i]}-{main_bins[i+1]}")
            
    return pd.cut(s, bins=main_bins, retbins=True, labels=labels)

def find_episodes(df:pd.DataFrame,target = TARGET_COLUMN, x = DATE_COLUMN, condition:Callable[[pd.Series], pd.Series] = (lambda x: x==1)) -> list[tuple]:
    condition_met = condition(df[target]).to_list()
    result = []

    start_index = None
    for i in range(len(condition_met)):
        if condition_met[i]:
            if start_index is None:
                start_index = i
        elif start_index is not None:
            start_x = df[x].iloc[start_index]
            end_x = df[x].iloc[i-1]
            result.append((start_x,end_x))
            start_index = None
    
    if start_index is not None:
        start_x = df[x].iloc[start_index]
        end_x = df[x].iloc[len(condition_met)-1]
        result.append((start_x,end_x))
    
    return result

if __name__ == "__main__":
    print(bin_data(pd.Series([1,2,3,4,5,6,7,8,9,10,11]),rmin=4,rmax=10))

