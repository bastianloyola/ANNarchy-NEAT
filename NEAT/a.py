def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Ejemplo con múltiples límites:
limits = [
    (-2.5, 2.5),
    (-2.5, 2.5),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-6.2831855, 6.2831855),
    (-10.0, 10.0),
    (-0.0, 1.0),
    (-0.0, 1.0)
]

# Valores de ejemplo para normalizar
values = [-1.25, 2.0, 0.0, -7.0, 3.14, 5.0, 0.5, 0.75]

normalized_values = [
    normalize(value, min_val, max_val)
    for value, (min_val, max_val) in zip(values, limits)
]

print(normalized_values)
