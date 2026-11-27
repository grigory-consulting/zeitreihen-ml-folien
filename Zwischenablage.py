def backtest(cols, label=y):
    results = []
    for train, test in fold_split:
        X_train, X_test = table.iloc[train][cols], table.iloc[test][cols]
        y_train, y_test = label.iloc[train][cols], label.iloc[test][cols]
        # Summe der positiven 
        n_pos = y_train.sum()
        # Logistische Regression
        logreg = make_pipeline(SimpleImputer(strategy = "median"),
                               StandardScaler(), # Normierung der Daten mu = 0, sigma = 1 
                                LogisticRegression(class_weight="balanced", max_iter=1000))
        logreg.fit(X_train, y_train) # Training
        # LightGBM 
        clf = LGBMClassifier(n_estimators = 300, learning_rate = 5e-2, num_leaves = 31,
                             scale_pos_weight = (len(y_train) - n_pos) / n_pos,
                             random_state = SEED, n_jobs = -1, verbose= -1)
        # Berechnet die Ereignisrate je Stunde des Tages im Training.
        rate_by_hour = y_train.groupby(table["hour"].iloc[train]).mean()
        # Speichert Labels, Scores, 0/1-Vorhersage und Modell des Folds.
        results.append({"y": y_test.to_numpy(), "p_model": clf.predict_proba(X_test)[:, 1],
                        "p_logreg": logreg.predict_proba(X_test)[:, 1],
                        "pred_model": clf.predict(X_test),
                        "p_hour": table["hour"].iloc[test].map(rate_by_hour).to_numpy(),
                        "p_strong": strong_baseline(train, test, y_train), "model": clf})
    # Gibt die Ergebnisse aller Folds zurück.
    return results

# Definiert die strengere Baseline.
def strong_baseline(train, test, y_train):
    """Strengere Baseline als Score für die Teststunden."""
    # Prüft, ob die freie Fassung gewählt ist.
    if AREA == "nordic":
        # Nutzt den Betrag von f_ext der Vorstunde als Score.
        return table["fext_abs_prev_1h"].iloc[test].fillna(0).to_numpy()  # Vorstunde ohne Messwert: Score 0
    # Berechnet die Ereignisrate je Stunde und Monat im Training.
    rate = y_train.groupby([table["hour"].iloc[train], table["month"].iloc[train]]).mean()
    # Ordnet jeder Teststunde die Rate ihrer Stunde und ihres Monats zu.
    return rate.reindex(pd.MultiIndex.from_frame(table[["hour", "month"]].iloc[test])).to_numpy()
