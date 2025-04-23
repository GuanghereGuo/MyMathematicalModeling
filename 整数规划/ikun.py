import numpy as np
import pulp

problem = pulp.LpProblem("ikun", pulp.LpMaximize)
vars = [
    pulp.LpVariable(f"x_{i}", lowBound=0, upBound=20, cat="Integer") for i in range(3)
]
c = np.array([20, 30, 40])
A_ub = np.array([4, 8, 10])
problem += pulp.lpSum(vars[i] * c[i] for i in range(3)), "Objective_Function"
problem += pulp.lpSum(vars[i] * A_ub[i] for i in range(3)) <= 100, "Weight_Constraint_1"
problem += pulp.lpSum(vars[i] for i in range(3)) <= 20, "Weight_Constraint_2"

problem.solve(pulp.PULP_CBC_CMD(msg=False))
print(problem.objective.value())
