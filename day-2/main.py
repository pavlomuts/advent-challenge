import polars as pl
from ortools.math_opt.python import mathopt
import networkx as nx

file_path = "day-2/instance.txt"
data = pl.read_csv(
    file_path,
    skip_rows=14,
    has_header=False,
    separator=" ",
    columns=[1, 2, 3, 4],
    new_columns=["node1", "node2", "distance", "cost"],
)

n = 100

graph = nx.DiGraph()
for row in data.iter_rows(named=True):
    graph.add_edge(
        row["node1"], row["node2"], distance=row["distance"], cost=row["cost"]
    )

graph.add_edge(100, 1, distance=0, cost=0)

model = mathopt.Model()

for edge in graph.edges():
    edge_dict = graph.edges[edge]
    x = model.add_binary_variable(name=f"{edge[0]}_{edge[1]}")
    edge_dict["x"] = x

for node in graph.nodes():
    model.add_linear_constraint(
        mathopt.fast_sum([graph[pred][node]["x"] for pred in graph.predecessors(node)])
        == mathopt.fast_sum([graph[node][succ]["x"] for succ in graph.successors(node)])
    )

model.add_linear_constraint(
    mathopt.fast_sum(
        graph.edges[edge]["cost"] * graph.edges[edge]["x"] for edge in graph.edges()
    )
    <= 73
)
model.add_linear_constraint(graph.edges[(100, 1)]["x"] == 1)

model.minimize(
    mathopt.fast_sum(
        graph.edges[edge]["distance"] * graph.edges[edge]["x"] for edge in graph.edges()
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
