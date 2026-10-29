"""A historical manual generator; it is not part of the runtime."""

import numpy as np

# Simulare de date
initial_values = np.linspace(7, 0, 120)  # Array descrescator de la 7 la 0
associated_percentages = np.random.uniform(-1.5, 1.5, 120)  # The associated percentages, between -150% and 150%.

# Aplicam formula de ajustare
adjusted_values = initial_values * np.minimum(2, np.maximum(0, 1 + associated_percentages))

# We print the first 10 values as an illustration.
#for i in range(10):
#   print(f"Initial value: {initial_values[i]:.2f}, Associated percentage: {associated_percentages[i]*100:.1f}%, Adjusted value: {adjusted_values[i]:.2f}")

# Simulare de date
initial_values = np.linspace(7, 0, 120)  # Array descrescator de la 7 la 0
associated_percentages = np.random.uniform(-1.0/10, 1.0/10, 120)  # The associated percentages, between -100% and 100%.

# Factor de scalare
n = len(initial_values)

# We apply the inverted exponential adjustment formula with bounds.
adjusted_values =  initial_values * np.minimum(2, np.maximum(0, 1 + np.exp(-np.arange(120) / n) * associated_percentages))

# We print the first 10 values as an illustration.
example_values_inverse_exp = [(initial_values[i], associated_percentages[i]*100, adjusted_values[i]) for i in range(120)]
for i in range(110):
    print(f"Initial value: {example_values_inverse_exp[i][0]:.2f}, Associated percentage: {example_values_inverse_exp[i][1]:.1f}%, Adjusted value: {example_values_inverse_exp[i][2]:.2f}")


