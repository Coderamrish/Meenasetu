import pandas as pd

df = pd.read_csv(
    "datasets/processed/FreshwaterfishdiversityofWestBengal_table_1.csv"
)

records = []

FAMILY_DEFAULTS = {
    "Bagridae": ("catfish", "carnivore", 22, 32, 6.5, 8.0, 3000),
    "Cyprinidae": ("carp", "omnivore", 20, 32, 6.5, 8.5, 4000),
    "Channidae": ("murrel", "carnivore", 22, 34, 6.0, 8.0, 2000),
    "Osphronemidae": ("gourami", "omnivore", 22, 30, 6.0, 7.5, 2500),
    "Clariidae": ("catfish", "carnivore", 24, 34, 6.0, 8.5, 3500)
}

for _, row in df.iterrows():
    sci = row["Order, Family, Species"]
    family = row["Order, Family, Species"].split(":")[-1].strip()

    if family not in FAMILY_DEFAULTS:
        continue

    category, feeding, tmin, tmax, pmin, pmax, density = FAMILY_DEFAULTS[family]

    records.append({
        "common_name": "",
        "scientific_name": sci,
        "family": family,
        "water_type": "pond",
        "region": "West Bengal",
        "season": "all_year",
        "temp_min": tmin,
        "temp_max": tmax,
        "ph_min": pmin,
        "ph_max": pmax,
        "stocking_density_per_ha": density,
        "growth_duration_months": 8,
        "category": category,
        "feeding_habit": feeding,
        "market_value": "medium",
        "is_native": True,
        "data_quality": "needs_verification"
    })

out = pd.DataFrame(records)
out.to_csv("data/final/fish_mapping_auto.csv", index=False)
print(f"Generated {len(out)} records")
