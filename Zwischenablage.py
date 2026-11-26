# Lab-Ordner finden (Jupyter, VS Code, PyCharm), damit `common` importierbar ist
import os
# Importiert sys für den Suchpfad der Module.
import sys
# Importiert warnings zum Filtern von Warnungen.
import warnings
# Importiert Path für Dateipfade.
from pathlib import Path

# Sucht den Lab-Ordner im Arbeitsordner und in den übergeordneten Ordnern.
for base in (Path.cwd(), Path.cwd() / "labs", *Path.cwd().parents):
    # Prüft, ob der Ordner das Paket common enthält.
    if (base / "common" / "__init__.py").exists():
        # Wechselt in den gefundenen Lab-Ordner.
        os.chdir(base)
        # Nimmt den Lab-Ordner in den Suchpfad für Importe auf.
        sys.path.insert(0, str(base))
        # Beendet die Suche nach dem ersten Treffer.
        break
warnings.filterwarnings("ignore", category=DeprecationWarning)  # numpy 2.5 meldet interne pandas-Aufrufe

# Importiert LightGBM für das Gradient Boosting.
import lightgbm as lgb
# Importiert matplotlib für die Grafiken.
import matplotlib.pyplot as plt
# Importiert numpy für Arrays.
import numpy as np
# Importiert pandas für Tabellen und Zeitreihen.
import pandas as pd

# Importiert Seed, Datenordner, Ladefunktion und MAE aus dem Kurspaket.
from common import SEED, find_data_dir, load, mae
# Importiert die Merkmals- und Backtest-Bausteine für Lab 2.
from common.lab2_features import (LOCAL_TZ, backtest, calendar_features, compare, day_ahead_folds,
                                  fold_mae, fourier_features, holiday_features, make_features)

# Gibt Arbeitsordner und LightGBM-Version aus.
print("Arbeitsordner:", Path.cwd().name, "| LightGBM", lgb.__version__)
