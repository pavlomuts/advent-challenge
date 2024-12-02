import polars as pl
from ortools.math_opt.python import mathopt

file_path = "day-1/instance.txt"
data = pl.read_csv(
    file_path,
    skip_rows=9,
    has_header=False,
    separator=" ",
    columns=[1, 2],
    new_columns=["event1", "event2"],
)

n = 100

# n events and at most n rooms.
# each event has to be assigned to a room.
# x_ij - event i is assigned to room j.

model = mathopt.Model()

variables = {}
for i in range(1, n + 1):
    for j in range(1, n + 1):
        variables[(i, j)] = model.add_binary_variable(name=f"x_{i}_{j}")

for i in range(1, n + 1):
    model.add_linear_constraint(
        mathopt.fast_sum([variables[(i, j)] for j in range(1, n + 1)]) == 1
    )

# r_j - room is occupied or not
# sum(x_ij) <= M * r_j

rooms = []
for j in range(n):
    rooms.append(model.add_binary_variable(name=f"r_{j}"))

for j in range(1, n + 1):
    model.add_linear_constraint(
        mathopt.fast_sum([variables[(i, j)] for i in range(1, n + 1)])
        <= n * rooms[j - 1]
    )

for j in range(1, n + 1):
    for row in data.iter_rows(named=True):
        k = row["event1"]
        m = row["event2"]
        model.add_linear_constraint(variables[(k, j)] + variables[m, j] <= 1)


model.minimize(mathopt.fast_sum(rooms))

result = mathopt.solve(
    model, mathopt.SolverType.HIGHS, params=mathopt.SolveParameters(enable_output=True)
)

print("Objective value:", result.objective_value())
