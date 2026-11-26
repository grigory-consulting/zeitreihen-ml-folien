# Legt die vier Verfahren für den Backtest an.
sf = StatsForecast(
    models=[SeasonalNaive(season_length=168), RandomWalkWithDrift(), AutoETS(season_length=24),
            MSTL(season_length=[24, 168])],
    freq="h",
    n_jobs=1,
)
# Legt die Spaltennamen der Verfahren fest.
MODELS = ["SeasonalNaive", "RWD", "AutoETS", "MSTL"]



# Startet die Zeitmessung.
start = time.perf_counter()
# Führt den Rolling-Origin-Backtest mit 14 Ursprüngen aus und ergänzt den Schritt.
cv = add_step(sf.cross_validation(df=df, h=24, step_size=24, n_windows=14, input_size=TRAIN_HOURS))
# Gibt Laufzeit und Bereich der Ursprünge aus.
print(f"Backtest: {time.perf_counter() - start:.1f} s, Ursprünge {cv['cutoff'].min()} bis {cv['cutoff'].max()} UTC")
# Legt MAE, Standardfehler, MASE und Anteil vor der Baseline je Verfahren nebeneinander.
score(cv, y, MODELS, TRAIN_HOURS)



# Berechnet den mittleren absoluten Fehler je Schritt nach dem Ursprung.
by_step = cv[MODELS ].sub(cv["y"], axis=0).abs().groupby(cv["step"]).mean()
# Zeichnet den Fehler je Schritt ohne Drift und AutoETS.
ax = by_step.drop(columns=["RWD", "AutoETS"]).plot(marker=".")
# Beschriftet die Achsen.
ax.set(xlabel="Schritt nach dem Ursprung (Stunden)", ylabel="MAE (MWh)", ylim=(0, None))
# Zeigt die Abbildung.
plt.show()
# Gibt Minimum und Maximum des Baseline-Fehlers je Schritt aus.
print(f"Baseline je Schritt: {by_step['SeasonalNaive'].min():.0f} bis {by_step['SeasonalNaive'].max():.0f} MWh")
