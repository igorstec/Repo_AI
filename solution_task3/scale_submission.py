import pandas as pd

# ── USTAW TUTAJ ──────────────────────────────────────────────────────────────
A = 1.1  # mnożnik dla maj, czerwiec, lipiec, sierpień (5, 6, 7, 8)
B = 0.9    # mnożnik dla wrzesień, październik (9, 10)

INPUT_CSV  = 'data/out/pliku.csv'
OUTPUT_CSV = 'data/out/pliku_scaled.csv'
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV)

df.loc[df['month'].isin([5, 6, 7, 8]), 'prediction'] *= A
df.loc[df['month'].isin([9, 10]),       'prediction'] *= B
df['prediction'] = df['prediction'].clip(0.0, 1.0)

df.to_csv(OUTPUT_CSV, index=False)

print(f'A={A} (maj-sie)  B={B} (wrz-paź)')
print(f'Saved: {OUTPUT_CSV}')
print(df.groupby('month')['prediction'].agg(['mean', 'min', 'max']).round(4))
