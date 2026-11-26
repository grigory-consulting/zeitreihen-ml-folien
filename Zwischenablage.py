# Baut Kalender- und Feiertagsmerkmale zum Zeitindex.
calendar = pd.concat([calendar_features(y.index), holiday_features(y.index)], axis=1)
# Legt vier Zeitpunkte um 12 Uhr Ortszeit fest.
noon = [pd.Timestamp(f"{d} 12:00", tz=LOCAL_TZ) for d in ("2026-06-03", "2026-06-04", "2026-11-01", "2025-12-29")]
# Gibt die Kalendermerkmale zu diesen Zeitpunkten aus.
print(calendar.loc[[t.tz_convert("UTC") for t in noon]].to_string())

# Fügt Lags, Rolling, Kalender und Fourier-Terme zur Merkmalstabelle zusammen.
X = pd.concat([lags, rolling, calendar, fourier_features(y.index)], axis=1)
pd.testing.assert_frame_equal(X, make_features(y))  # dieselbe Tabelle wie in der Übung
# Gibt Größe und erste Spalten der Merkmalstabelle aus.
print("Merkmalstabelle:", X.shape, "| Spalten:", ", ".join(X.columns[:10]), "...")



def make_lgbm():
    # Gibt ein neues Modell mit festem Seed zurück.
    return lgb.LGBMRegressor(random_state=SEED, verbose=-1)


# Führt den Rolling-Origin-Backtest mit LightGBM aus.
res = backtest(make_lgbm, X, y).rename(columns={"pred": "lightgbm"})
# Ergänzt Seasonal Naive (Lag 168) als Vergleich.
res["seasonal_naive"] = y.shift(168).reindex(res.index)
# Gibt Testzeitraum und Zahl der Teststunden aus.
print("Testzeitraum:", res.index.min(), "bis", res.index.max(), "|", len(res), "Stunden")
# Vergleicht beide Prognosen in einer Tabelle.
compare(res, y, ["seasonal_naive", "lightgbm"])



res["published"] = data["load_forecast_mwh"].reindex(res.index)
# Vergleicht Seasonal Naive, veröffentlichte Prognose und LightGBM.
table = compare(res, y, ["seasonal_naive", "published", "lightgbm"])
# Zählt die Folds, in denen LightGBM den kleineren MAE hat.
wins = int((fold_mae(res, "lightgbm") < fold_mae(res, "published")).sum())
# Berechnet den relativen Unterschied im MAE in Prozent.
gap = 100 * (table.loc["lightgbm", "MAE"] / table.loc["published", "MAE"] - 1)
# Gibt Unterschied und Zahl der gewonnenen Folds aus.
print(f"LightGBM gegen veröffentlicht: {gap:+.1f} % MAE, besser in {wins} von 6 Folds")
# Zeigt die Vergleichstabelle.
table
