import pulp
import numpy as np

# Define coordinates
yard_A = (5, 1)
yard_B = (2, 7)
sites = [(1.25, 1.25), (8.75, 0.75), (0.5, 4.75), (5.75, 5), (3, 6.5), (7.25, 7.25)]
demands = [3, 5, 4, 7, 6, 11]

# Function to calculate Euclidean distance
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Calculate distances from A and B to each site
d_A = [distance(yard_A, site) for site in sites]
d_B = [distance(yard_B, site) for site in sites]

# Create the LP problem
prob = pulp.LpProblem("Minimize_Ton_Kilometers", pulp.LpMinimize)

# Define decision variables
x = [pulp.LpVariable(f"x_{j+1}", lowBound=0) for j in range(6)]  # From A to site j
y = [pulp.LpVariable(f"y_{j+1}", lowBound=0) for j in range(6)]  # From B to site j

# Set the objective function
prob += sum(x[j] * d_A[j] + y[j] * d_B[j] for j in range(6)), "Total_Ton_Kilometers"
prob += sum(x[j] * d_A[j] + y[j] * d_B[j] for j in range(6)), "Total_Ton_Kilometers"

# Add demand constraints
for j in range(6):
    prob += (x[j] + y[j] == demands[j]), f"Demand_Site_{j+1}"

# Solve the problem
prob.solve()

# Check solver status
status = pulp.LpStatus[prob.status]
print(f"Solver Status: {status}")

# Output the optimal supply plan
print("Optimal Daily Supply Plan:")
for j in range(6):
    from_A = x[j].varValue
    from_B = y[j].varValue
    site_name = ["I", "II", "III", "IV", "V", "VI"][j]
    if from_A > 0:
        print(f"From Yard A to Site {site_name}: {from_A} tons")
    if from_B > 0:
        print(f"From Yard B to Site {site_name}: {from_B} tons")

# Output the minimal total ton-kilometers
total_ton_km = pulp.value(prob.objective)
print(f"Minimal Total Ton-Kilometers: {total_ton_km:.3f}")