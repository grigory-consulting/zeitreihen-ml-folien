# Legt drei übereinanderliegende Achsen an.
fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
# Zeichnet Beobachtung, Trend und Rest je in eine Achse.
for ax, (label, part) in zip(axes, [("Beobachtung", year), ("Trend", res.trend), ("Rest", res.resid)]):
    # Zeichnet die Komponente als Linie.
    ax.plot(part.index, part.values, lw=0.5)
    # Beschriftet die Achse mit dem Namen der Komponente.
    ax.set_ylabel(label)
# Ordnet die Achsen ohne Überlappung an.
plt.tight_layout()
# Zeigt die Abbildung.
plt.show()

# Legt eine neue Achse an.
fig, ax = plt.subplots()
# Zeichnet beide Saisonkomponenten über zwei Märzwochen.
res.seasonal["2025-03-03":"2025-03-16"].plot(ax=ax, lw=1)
# Beschriftet Achsen und Titel.
ax.set(ylabel="MWh", xlabel="", title="Saisonkomponenten, zwei Wochen im März 2025")
# Zeigt die Abbildung.
plt.show()



# Mittelt den Rest je Kalendertag in Ortszeit.
daily_resid = res.resid.tz_convert("Europe/Berlin").resample("D").mean()
# Lädt die bundesweiten Feiertage 2025.
national = holidays.Germany(years=2025)
# Lädt die Feiertage 2025 in Baden-Württemberg.
bw = holidays.Germany(subdiv="BW", years=2025)


# Definiert eine Funktion, die zu einem Tag den Feiertagsnamen liefert.
def holiday_label(day):
    # Prüft, ob der Tag bundesweit Feiertag ist.
    if day in national:
        # Gibt den bundesweiten Feiertagsnamen zurück.
        return national[day]
    # Gibt den Landesfeiertag mit Zusatz oder einen leeren Text zurück.
    return f"{bw[day]} (nicht bundesweit)" if day in bw else ""


# Wählt die zehn Tage mit dem größten Betrag des Rests.
top = daily_resid.abs().sort_values(ascending=False).head(10).index
# Legt die Kürzel der Wochentage fest.
DAYS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
# Gibt Wochentag, mittleren Rest und Feiertag der zehn Tage als Tabelle aus.
print(pd.DataFrame({"Wochentag": [DAYS[d] for d in top.dayofweek], "mittlerer Rest (MWh)": daily_resid[top].round(0).values,
                    "Feiertag": [holiday_label(d.date()) for d in top]}, index=top.date))
# Gibt die Streuung von Rest und beiden Saisonkomponenten aus.
print(f"Streuung des Rests: {res.resid.std():.0f} MWh, Saison 24: {res.seasonal['seasonal_24'].std():.0f}, "
      f"Saison 168: {res.seasonal['seasonal_168'].std():.0f}")
