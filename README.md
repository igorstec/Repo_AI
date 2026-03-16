***

## `prev_optimized.py` — opis rozwiązania

Pipeline do predykcji zużycia energii (`x2`) per urządzenie i miesiąc, oparty na LightGBM z osobnymi modelami per typ urządzenia.

***

### Architektura

```
data3.csv → Feature Engineering → LightGBM (per deviceType) → Bias Correction → Monthly Aggregation → submission.csv
```

***

### Feature Engineering (`main()`)

**Cechy czasowe**
- Wyciąga `hour`, `dayofweek`, `month`, `day`, `year` z timestamp
- Kodowanie cykliczne (sin/cos) dla godziny, miesiąca i dnia tygodnia — model wie, że 23:00 ≈ 0:00
- Flagi binarne: `is_night` (godz. 0–6 i 22–23), `is_weekend` (sob–nie)

**Cechy temperaturowe**
- Różnice: zewnętrzna–wewnętrzna, wymienniki load/source, odczuwalna–faktyczna
- `heating_demand` — ile temperatury brakuje do progu komfortu (0.7)
- `cooling_demand` — ile powyżej progu komfortu
- `hex_cross = (t3−t5)×(t4−t6)` — interakcja wymienników source×load
- Nieliniowości: `temp²`, reżim ciepły (>10°C), zimny (<10°C), `temp×is_night`

**Cechy urządzenia**
- One-hot encoding `deviceType` (7, 11, 19) + interakcja `temperatura × typ`
- Per-device statystyki z danych treningowych: mean/std/median/min/max dla `x2`
- Per-deviceType statystyki: mean/std dla `x2`
- Per-device profil godzinowy: średnie `x2` per `device × hour`
- Per-device wrażliwość na temperaturę: slope regresji liniowej `temp→x2` (ogólny + dla reżimu >10°C)

**Rolling features** (`add_rolling_features`, linia 30–49)
- Liczone osobno dla każdego urządzenia (groupby deviceId)
- Okna: **6h** i **24h** dla `temperature`, `t1`, `t2`, `t10`, `x1` → średnia krocząca
- Std temperatury z 24h (zmienność pogody)
- Wyłącznie `.rolling()` — tylko dane wsteczne, **zero data leakage**

***

### Trenowanie

Osobny model LightGBM dla każdego z 3 typów urządzeń (7, 11, 19).

| Krok | Opis |
|---|---|
| **OOF (5-fold CV)** | 5 modeli, każdy predykuje na własnym holdout fold — predykcje bez leakage na całym train |
| **Final model** | Trenuje na 100% danych danego typu |
| **Predict** | Predykcja na valid + test, clip do ≥ 0 |

**Sample weights (`[4]`):**
- Kwiecień i Październik: waga `×2.0` (przejściowe miesiące, najbliższe oknu predykcji maj–październik)
- Miesiące zimowe: waga `×0.7`

***

### Bias Correction

Koryguje systematyczne błędy per urządzenie:

```
oof_residual = x2 - oof_pred
dev_bias = mean(oof_residual) per device
final_pred += dev_bias
```

Jeśli model systematycznie zaniża urządzenie X o 0.02, dodaje 0.02 do wszystkich jego predykcji. Skuteczne dla urządzeń z unikalnym profilem zużycia.

***

### Agregacja i walidacja

- Uśrednia predykcje godzinowe → miesięczne per urządzenie
- Brakujące kombinacje `device × month` uzupełnia średnią dla danego urządzenia
- Walidacja: OOF monthly MAE na treningu, osobno dla Kwietnia

***

### Dane wejściowe / wyjściowe

| Plik | Opis |
|---|---|
| `data3.csv` | Dane wejściowe (hourly, timestamp + deviceId + sensory) |
| `submission.csv` | Predykcje monthly per device |

![Task3](./task3.jpg)
![Task3](./task32.jpg)
![Task3](./task33.jpg)
![Task3](./task34.jpg)
