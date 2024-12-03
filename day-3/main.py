import polars as pl
from ortools.math_opt.python import mathopt

n = 100

file_path = "day-3/instance.txt"
data = pl.read_csv(
    file_path,
    skip_rows=11,
    has_header=False,
    separator=" ",
    new_columns=[str(item) for item in range(n)],
)

model = mathopt.Model()

variables = {}
for i in range(n):
    for j in range(n):
        variables[(i, j)] = model.add_binary_variable(name=f"x_{i}_{j}")

for i in range(n):
    model.add_linear_constraint(
        mathopt.fast_sum([variables[(i, j)] for j in range(n)]) == 1
    )

model.minimize(
    mathopt.fast_sum(
        data.item(i, j) * variables[(i, j)] for i in range(n) for j in range(n)
    )
)

result = mathopt.solve(
    model, mathopt.SolverType.HIGHS, params=mathopt.SolveParameters(enable_output=True)
)

print("Objective value:", result.objective_value())

for v in model.variables():
    x = result.variable_values()[v]
    if x == 1:
        print(v.name)