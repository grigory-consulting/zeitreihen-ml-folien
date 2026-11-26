# Legt die vier Verfahren für den Backtest an.
sf = StatsForecast(
    models=[SeasonalNaive(season_length=168), RandomWalkWithDrift(), AutoETS(season_length=24),
            MSTL(season_length=[24, 168])],
    freq="h",
    n_jobs=1,
)
# Legt die Spaltennamen der Verfahren fest.
MODELS = ["SeasonalNaive", "RWD", "AutoETS", "MSTL"]
