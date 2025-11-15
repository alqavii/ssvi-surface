from models.options_data import OptionsModel
from models.config_model import IVConfig
import pandas as pd
from typing import List


class IVEngine:
    @staticmethod
    def generateIVSmile(
        cfg: IVConfig,
        options: List[OptionsModel],
        rate: float,
        spot: float,
    ) -> pd.DataFrame:
        calls = [opt for opt in options]

        return pd.DataFrame()
