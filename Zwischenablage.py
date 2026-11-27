# Legt das Gebiet fest, aus dem das Lab die Daten liest.
AREA = "ce"  # "ce": Kontinentaleuropa (Kursfassung), "nordic": Nordic (freie Fassung)

# Importiert das Modul für Betriebssystemfunktionen.
import os
# Importiert das Modul für den Suchpfad der Imports.
import sys
# Importiert die Klasse für Dateipfade.
from pathlib import Path

# Sucht den Ordner labs, der das Paket common enthält.
LABS = next(p for p in (Path.cwd(), Path.cwd() / "labs", Path.cwd().parent) if (p / "common").is_dir())
# Wechselt in den Ordner labs.
os.chdir(LABS)
# Setzt den Ordner labs an den Anfang des Suchpfads.
sys.path.insert(0, str(LABS))

# Importiert matplotlib für die Abbildungen.
import matplotlib.pyplot as plt
# Importiert NumPy für Arrays.
import numpy as np
# Importiert pandas für Tabellen.
import pandas as pd
# Importiert den LightGBM-Klassifikator.
from lightgbm import LGBMClassifier
# Importiert den Imputer, der Fehlwerte füllt.
from sklearn.impute import SimpleImputer
# Importiert die logistische Regression.
from sklearn.linear_model import LogisticRegression
# Importiert Average Precision und Precision-Recall-Kurve.
from sklearn.metrics import average_precision_score, precision_recall_curve
# Importiert die Funktion, die Schritte zu einer Pipeline verkettet.
from sklearn.pipeline import make_pipeline
# Importiert den Standardisierer.
from sklearn.preprocessing import StandardScaler

# Importiert Seed, Gebietsdaten und Ereignisschwelle aus dem Kurspaket.
from common import SEED, area_config, default_threshold
# Importiert Entscheidungszeit, Label, Merkmalsätze, Folds und Stundentabelle.
from common.lab4_events import DECISION_TIME, event_label, feature_sets, folds, hourly_table

# Stellt die Ausgabebreite von pandas ein.
pd.set_option("display.width", 120)
# Gibt den Namen des gewählten Gebiets aus.
print(f"Gebiet: {area_config(AREA)['name']}")
