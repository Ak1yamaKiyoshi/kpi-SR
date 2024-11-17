import numpy as np
from tabulate import tabulate  # If you don't have tabulate, install it with: pip install tabulate

def evaluate_sturm_sequence(x):
    f0 = 2*x**5 - 2*x**4 + 5*x**3 + x**2 - 2*x
    f1 = 10*x**4 - 8*x**3 + 15*x**2 + 2*x - 2
    f2 = -4*x**3 - 30*x**2 + 38*x + 2
    f3 = -1465
    return [f0, f1, f2, f3]

def get_sign(value):
    if abs(value) < 1e-10:  # Handle near-zero values
        return '0'
    return '+' if value > 0 else '-'

def count_sign_changes(values):
    signs = [get_sign(v) for v in values]
    changes = 0
    for i in range(len(signs)-1):
        if signs[i] != signs[i+1] and signs[i] != '0' and signs[i+1] != '0':
            changes += 1
    return changes

# Ranges to check
ranges = [
    (-1.5, -1),
    (-1, -0.5),
    (-0.5, 0),
    (0, 0.5),
    (0.5, 1)
]

# Prepare table data
headers = ["Range", "f₀(x)", "f₁(x)", "f₂(x)", "f₃(x)", "Sign Changes"]
table_data = []

for start, end in ranges:
    mid = (start + end) / 2
    values = evaluate_sturm_sequence(mid)
    signs = [get_sign(v) for v in values]
    changes = count_sign_changes(values)
    
    # Format range with proper spacing
    range_str = f"({start:>4}, {end:<4})"
    table_data.append([range_str] + signs + [changes])

# Print fancy table
print("\nSturm's Theorem Sign Analysis")
print("=" * 60)
print(tabulate(table_data, headers=headers, tablefmt="pretty"))
print("\nFunctions:")
print(f"f₀(x) = 2x⁵ - 2x⁴ + 5x³ + x² - 2x")
print(f"f₁(x) = 10x⁴ - 8x³ + 15x² + 2x - 2")
print(f"f₂(x) = -4x³ - 30x² + 38x + 2")
print(f"f₃(x) = -1465")