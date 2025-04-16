from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ToyDataset:
    def __init__(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        label: Union[pd.DataFrame, np.ndarray]
    ) -> None:
