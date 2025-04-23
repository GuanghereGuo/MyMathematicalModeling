import numpy as np
import pulp

t, m = map(int, input().split())

items = np.array([list(map(int, input().split())) for _ in range(m)])

# 创建线性规划问题
problem = pulp.LpProblem(name="01", sense=pulp.LpMaximize)
# 创建决策变量
vars = [pulp.LpVariable(name=f"x_{i}", lowBound=0, upBound=1, cat="Integer") for i in range(m)]

problem += pulp.lpSum(vars[i] * items[i][0] for i in range(m)) <= t, "Weight_Constraint"
problem += pulp.lpSum(vars[i] * items[i][1] for i in range(m)), "Objective_Function"

problem.solve(pulp.PULP_CBC_CMD(msg=False))

print(int(problem.objective.value()))
