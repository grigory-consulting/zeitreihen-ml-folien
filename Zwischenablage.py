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
