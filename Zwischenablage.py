# einfache Baseline Prognose trainieren
df = common.to_nixtla(y.to_frame("DE"))
TRAIN_HOURS = 24 * 7 * 8 # acht Wochen 
history, future = df.iloc[:-24], df.iloc[-24:]
# Seasonal Naive mit Lag 168 und Naive mit Drift 
sf_simple = StatsForecast(models = [SeasonalNaive(season_length = 168), RandomWalkWithDrift()], 
                          freq="h", n_jobs = -1)
# Prognostiziert 24 Stunden und legt die Istwerte daneben.
fc = sf_simple.forecast(df=history.tail(TRAIN_HOURS), h=24).merge(future, on=["unique_id", "ds"])

# Zeichnet die letzten 72 Stunden des Verlaufs.
ax = history.tail(72).plot(x="ds", y="y", label="Verlauf", color="grey")
# Zeichnet Ist und beide Prognosen in dieselbe Achse.
fc.plot(x="ds", y=["y", "SeasonalNaive", "RWD"], ax=ax, label=["Ist", "Seasonal Naive", "Naive mit Drift"])
# Beschriftet die Achsen.
ax.set(xlabel="UTC", ylabel="MWh")
# Zeigt die Abbildung.
plt.show()
# Geht beide Verfahren durch.
for model in ["SeasonalNaive", "RWD"]:
    # Gibt den MAE des Verfahrens aus.
    print(f"{model:14s} MAE {common.mae(fc['y'], fc[model]):6.0f} MWh")
