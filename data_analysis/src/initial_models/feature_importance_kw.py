# misc: cols 0-1
# numericals: cols 2-11
# binaries: cols 12-24 (includes Stage_nan)
# target: col 25

import numpy as np
import pandas as pd

df = pd.read_csv('data/cirrhosis-CLEAN-kw.csv')