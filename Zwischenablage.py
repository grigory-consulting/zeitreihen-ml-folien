# Wählt das Netzgebiet für das Lab.
AREA = "ce"

# Importiert os für den Wechsel des Arbeitsverzeichnisses.
import os
# Importiert sys für den Suchpfad der Module.
import sys
# Importiert warnings zum Ausblenden von Hinweisen.
import warnings
# Importiert Path für Dateipfade.
from pathlib import Path

# Sucht den Lab-Ordner mit dem Paket common.
LABS = next(p for p in (Path.cwd(), Path.cwd() / "labs", Path.cwd().parent) if (p / "common").is_dir())
# Wechselt in den Lab-Ordner.
os.chdir(LABS)
# Macht das Paket common importierbar.
sys.path.insert(0, str(LABS))

# Importiert matplotlib für die Grafiken.
import matplotlib.pyplot as plt
# Importiert numpy für Berechnungen auf Arrays.
import numpy as np
# Importiert pandas für Zeitreihen und Tabellen.
import pandas as pd
# Importiert den Isolation Forest aus scikit-learn.
from sklearn.ensemble import IsolationForest

# Blendet DeprecationWarnings aus.
warnings.filterwarnings("ignore", category=DeprecationWarning)   # Hinweise aus numpy/pandas, ohne Folgen hier

# Importiert die gemeinsamen Hilfsfunktionen der Labs.
import common
# Importiert Ereignisbildung, Ereignisbewertung und Regler.
from common import build_events, event_scores, explore
# Importiert CUSUM, Zeiträume und Schwellensuche für Lab 3.
from common.lab3_detectors import cusum, periods, threshold_for_budget

# Liest die Einstellungen des Gebiets.
cfg = common.area_config(AREA)
# Gibt Gebiet, Schwelle und Seed aus.
print(f"Gebiet: {cfg['name']}, Schwelle {common.default_threshold(AREA) * 1000:.0f} mHz, Seed {common.SEED}")
# Gibt den Zeitpunkt des bekannten Ereignisses aus.
print(f"Bekanntes Ereignis: {cfg['known_event']:%d.%m.%Y %H:%M:%S} UTC")
