# Holt die Trainingsmaske des letzten Folds.
train, _ = list(day_ahead_folds(X.index, 6, 56))[-1]
# Trainiert das Modell auf diesem Fold.
model = make_lgbm().fit(X[train], y[train])
# Liest den Gain je Merkmal aus dem Booster.
gain = pd.Series(model.booster_.feature_importance("gain"), index=X.columns)
# Wandelt den Gain in Prozentanteile um und sortiert sie.
share = (100 * gain / gain.sum()).sort_values()
# Zeichnet die zwölf größten Anteile als Balken.
ax = share.tail(12).plot.barh(figsize=(7, 3.8))
# Beschriftet Achse und Titel.
ax.set(xlabel="Anteil am Gain in %", title="Merkmalswichtigkeit, letzter Fold")
# Passt die Ränder der Grafik an.
plt.tight_layout()
# Zeigt die Grafik.
plt.show()
# Gibt die fünf größten Anteile aus.
print(share.sort_values(ascending=False).head(5).round(1).to_string())
