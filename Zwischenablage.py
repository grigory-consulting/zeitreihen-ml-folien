# Importiert die Standardmodule für Pfade, Zeitmessung und Warnungen.
import os
import sys
import time
import warnings
from pathlib import Path

# Sucht den Lab-Ordner, in dem das Paket common liegt.
LABS = next(p for p in (Path.cwd(), Path.cwd() / "labs", Path.cwd().parent) if (p / "common").is_dir())
# Wechselt in den Lab-Ordner.
os.chdir(LABS)
# Macht das Paket common importierbar.
sys.path.insert(0, str(LABS))
# Blendet DeprecationWarnings aus.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importiert Feiertage, Plot, pandas und die Prognoseverfahren.
import holidays
import matplotlib.pyplot as plt
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS, RandomWalkWithDrift, SeasonalNaive
from statsmodels.tsa.seasonal import MSTL as MSTLDecomposition

# Importiert die Lab-Helfer für Laden, Backtest-Schritt und Score.
import common
from common.lab1_backtest import add_step, score

# Legt die Standardgröße der Abbildungen fest.
plt.rcParams["figure.figsize"] = (11, 3.2)
