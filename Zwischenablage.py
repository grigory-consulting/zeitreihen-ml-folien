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
