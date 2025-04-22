import pulp
import numpy as np
from scipy.optimize import minimize

# Define site coordinates and demands
sites = [(1.25, 1.25), (8.75, 0.75), (0.5, 4.75), (5.75, 5), (3, 6.5), (7.25, 7.25)]
demands = [3, 5, 4, 7, 6, 11]
num_sites = len(sites)


# Function to calculate Euclidean distance
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Inner LP problem: Minimize ton-kilometers for fixed yard locations
def solve_transportation_problem(x_C, y_C, x_D, y_D):
    yard_C = (x_C, y_C)
    yard_D = (x_D, y_D)

    # Calculate distances
    d_C = [distance(yard_C, site) for site in sites]
    d_D = [distance(yard_D, site) for site in sites]

    # Create the LP problem
    prob = pulp.LpProblem("Minimize_Ton_Kilometers", pulp.LpMinimize)

    # Define decision variables
    u = [pulp.LpVariable(f"u_{j + 1}", lowBound=0) for j in range(num_sites)]  # From C to site j
    v = [pulp.LpVariable(f"v_{j + 1}", lowBound=0) for j in range(num_sites)]  # From D to site j

    # Objective function
    prob += sum(u[j] * d_C[j] + v[j] * d_D[j] for j in range(num_sites))

    # Demand constraints
    for j in range(num_sites):
        prob += (u[j] + v[j] == demands[j]), f"Demand_Site_{j + 1}"

    # Supply constraints
    prob += sum(u[j] for j in range(num_sites)) <= 20, "Supply_C"
    prob += sum(v[j] for j in range(num_sites)) <= 20, "Supply_D"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Return the objective value and the solution
    total_ton_km = pulp.value(prob.objective)
    u_vals = [pulp.value(u[j]) for j in range(num_sites)]
    v_vals = [pulp.value(v[j]) for j in range(num_sites)]
    return total_ton_km, u_vals, v_vals


# Objective function for the outer optimization (yard locations)
def objective(coords):
    x_C, y_C, x_D, y_D = coords
    total_ton_km, _, _ = solve_transportation_problem(x_C, y_C, x_D, y_D)
    return total_ton_km


# Initial guess for yard locations (e.g., average of site coordinates)
avg_x = np.mean([site[0] for site in sites])
avg_y = np.mean([site[1] for site in sites])
initial_guess = [avg_x - 1, avg_y - 1, avg_x + 1, avg_y + 1]  # Slightly offset for C and D

# Bounds for coordinates (optional, based on site ranges)
x_bounds = (min(site[0] for site in sites) - 2, max(site[0] for site in sites) + 2)
y_bounds = (min(site[1] for site in sites) - 2, max(site[1] for site in sites) + 2)
bounds = [x_bounds, y_bounds, x_bounds, y_bounds]

# Optimize yard locations
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, options={'disp': False})

# Get the optimal yard locations
x_C_opt, y_C_opt, x_D_opt, y_D_opt = result.x

# Solve the transportation problem with the optimal yard locations
total_ton_km_opt, u_vals_opt, v_vals_opt = solve_transportation_problem(x_C_opt, y_C_opt, x_D_opt, y_D_opt)

# Output the results
print("Optimal Locations for New Yards:")
print(f"Yard C: ({x_C_opt:.3f}, {y_C_opt:.3f})")
print(f"Yard D: ({x_D_opt:.3f}, {y_D_opt:.3f})")
print("\nOptimal Daily Supply Plan:")
for j in range(num_sites):
    u_val = u_vals_opt[j]
    v_val = v_vals_opt[j]
    site_name = ["I", "II", "III", "IV", "V", "VI"][j]
    if u_val > 0:
        print(f"From Yard C to Site {site_name}: {u_val:.3f} tons")
    if v_val > 0:
        print(f"From Yard D to Site {site_name}: {v_val:.3f} tons")
print(f"\nMinimal Total Ton-Kilometers: {total_ton_km_opt:.3f}")
print(f"Total Supply from Yard C: {sum(u_vals_opt):.3f} tons")
print(f"Total Supply from Yard D: {sum(v_vals_opt):.3f} tons")