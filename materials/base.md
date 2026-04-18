# Material Reference Values

Thermal properties at ~20 °C (293 K) for use in `scripts/calculate.py`.

Set `k`, `rho`, and `cp` — `alpha` is computed automatically.

| Material        | k [W/(m·K)] | rho [kg/m³] | cp [J/(kg·K)] | alpha [m²/s]       |
|-----------------|-------------|-------------|---------------|--------------------|
| Steel (carbon)  | 50          | 7800        | 500           | 1.28 × 10⁻⁵        |
| Stainless steel | 16          | 7900        | 500           | 4.05 × 10⁻⁶        |
| Aluminum        | 205         | 2700        | 900           | 8.44 × 10⁻⁵        |
| Copper          | 400         | 8900        | 385           | 1.17 × 10⁻⁴        |
| Gold            | 315         | 19300       | 129           | 1.27 × 10⁻⁴        |
| Titanium        | 22          | 4500        | 520           | 9.40 × 10⁻⁶        |
| Cast iron       | 50          | 7200        | 460           | 1.51 × 10⁻⁵        |
| GFRP            | 0.35        | 1850        | 1200          | 1.57 × 10⁻⁷        |
| Concrete        | 1.7         | 2300        | 880           | 8.40 × 10⁻⁷        |
| Glass           | 1.0         | 2500        | 750           | 5.33 × 10⁻⁷        |
| Wood (oak)      | 0.17        | 700         | 1700          | 1.43 × 10⁻⁷        |

## Usage

```python
# GFRP (glass fiber reinforced plastic)
k   = 0.35
rho = 1850.0
cp  = 1200.0

# Steel (carbon)
k   = 50.0
rho = 7800.0
cp  = 500.0

# Aluminum
k   = 205.0
rho = 2700.0
cp  = 900.0

# Copper
k   = 400.0
rho = 8900.0
cp  = 385.0

alpha = k / (rho * cp)  # computed automatically in the script
```

## Notes

- Properties vary with temperature; values here are for ambient conditions (~20 °C).
- For high-temperature simulations (e.g. fresh from an oven at 500 K+), consider using
  temperature-dependent properties `k(T)`, `rho(T)`, `cp(T)`.
- Sources: Engineering Toolbox, NIST, Incropera "Fundamentals of Heat and Mass Transfer".
