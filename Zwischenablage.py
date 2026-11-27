# Teilt die Zeit in Rolling-Origin-Folds mit Gap.
fold_list = folds(AREA, y)

# Legt eine leere Liste für die Zeilen der Übersicht an.
rows = []
# Geht die Folds der Reihe nach durch.
for k, (train, test) in enumerate(fold_list, start=1):
    # Wählt die Testlabels des Folds.
    y_test = y.iloc[test]
    # Hängt Zeitraum, Größe und Prävalenz des Folds als Zeile an.
    rows.append({"Fold": k, "Training bis": y.index[train[-1]].date(), "Test ab": y.index[test[0]].date(),
                 "Test bis": y.index[test[-1]].date(), "Stunden Test": len(test),
                 "positiv": int(y_test.sum()), "Prävalenz": round(y_test.mean(), 4)})
# Zeigt die Übersicht als Tabelle je Fold.
pd.DataFrame(rows).set_index("Fold")
