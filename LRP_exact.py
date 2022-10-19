import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd
import time
# input parameters

depots_fixedCost = pd.read_excel("Depots_Parameters.xlsx", sheet_name="Fixed_cost", index_col=0) #opening cost
depots_fixedCost = depots_fixedCost.to_numpy()

depots_inboundCost = pd.read_excel("Depots_Parameters.xlsx", sheet_name="Inbound_cost", index_col=0) #inbound cost
depots_inboundCost = depots_inboundCost.to_numpy()

depots_ordercost = pd.read_excel("Depots_Parameters.xlsx", sheet_name="Picking_cost", index_col=0) #processing cost per order at each location
depots_ordercost = depots_ordercost.to_numpy()

depots_capacity = pd.read_excel("Depots_Parameters.xlsx", sheet_name="Capacity", index_col=0) #number of customers per facility
num_capacity = len(depots_capacity.columns)
depots_capacity = depots_capacity.to_numpy()

depots_coords = pd.read_excel("Depots_Parameters.xlsx", sheet_name="Coords", index_col=0)
num_locations = len(depots_coords)

customers = pd.read_excel("Coordinates_Customers.xlsx", sheet_name="Exact_5",index_col=0)
num_customers = len(customers)

vehicles = len(customers)
vehicle_capacity = 5

all_coords = np.vstack((depots_coords, customers))
num_coords = len(all_coords)
distance = distance_matrix(all_coords, all_coords) #distance matrix with Euclidean distances
cost_per_distance = 1

shipping_cost =cost_per_distance*distance

start_time_total = time.process_time()

m = gp.Model("LRP_exact")


# Create variables
y = {} #if depot j and location i is open or not
for i in range(num_locations):
    for j in range(num_capacity):
        y[i,j] = m.addVar(vtype=GRB.BINARY, obj=depots_fixedCost[i][j]+depots_inboundCost[i][j], name='y(' + str(i) + ',' + str(j) + ')')

z = {} #if customer k is assigned to location j or not
for i in range(num_locations, num_locations+num_customers):
    for j in range(num_locations):
        for k in range(num_capacity):
            z[i,j,k] = m.addVar(vtype=GRB.BINARY, obj=depots_ordercost[j][k], name='z(' + str(i) + ',' + str(j)  + ',' + str(k)+ ')')

x = {} #if vehicle k travels from i to j
for i in range(num_customers + num_locations):
    for j in range(num_customers + num_locations):
        for k in range(vehicles):
            x[i,j,k] = m.addVar(vtype=GRB.BINARY, obj=shipping_cost[i][j], name='x(' + str(i) + ',' + str(j)  + ',' + str(k)+ ')')

s = {} #for subtour elimination
for i in range(num_locations, num_locations+num_customers):
    s[i] = m.addVar()

# each customer is assigned to 1 tour
for i in range(num_locations, num_locations+num_customers):
    m.addConstr(sum(x[i,j,k] for j in range(num_locations+num_customers) for k in range(vehicles)) == 1)

# at each location only 1 depot can be built
for i in range(num_locations):
    m.addConstr(sum(y[i,j] for j in range(num_capacity)) <= 1)

#each vehicle starts at most from one depot (single tour)
for k in range(vehicles):
    m.addConstr(sum(x[i,j,k] for i in range(num_locations) for j in range(num_locations, num_locations+num_customers)) <= 1)

#each vehicle tour includes a depot
m.addConstr(sum(x[i,j,k] for i in range(num_locations) for j in range(num_locations+num_customers) for k in range(vehicles)) >= 1)

#each vehicle that delivers to a customer/depot also leaves the customer/depot
for i in range(num_locations+num_customers):
    for k in range(vehicles):
        m.addConstr(sum(x[i,j,k] for j in range(num_locations+num_customers)) == sum(x[j,i,k] for j in range(num_locations+num_customers)))

#depot and customer have to be on the same route if customer allocated to depot
for i in range(num_locations, num_locations+num_customers):
    for j in range(num_locations):
        for k in range(vehicles):
            m.addConstr((sum(x[i,h,k] for h in range(num_locations+num_customers)) + sum(x[j,h,k] for h in range(num_locations+num_customers))) <= (1 + sum(z[i,j,n] for n in range(num_capacity))))

#customer demand allocated to each depot is less or equal to the capacity of the depot and customer can only be allocated to an open depot
for j in range(num_locations):
    for k in range(num_capacity):
        m.addConstr(sum(z[i,j,k] for i in range(num_locations, num_locations+num_customers)) <= depots_capacity[j][k] * y[j,k])

#max tour stops
for k in range(vehicles):
    m.addConstr(sum(x[i,j,k] for i in range(num_locations, num_locations+num_customers) for j in range(num_customers + num_locations)) <= vehicle_capacity)
    # m.addConstr(sum(distance[i, j] * x[i, j, k] for i in range(num_customers) for j in range(num_customers)) <= time)

#Subtourelimination
for i in range(num_locations, num_locations+num_customers):
    for j in range(num_locations, num_locations+num_customers):
        if i != j:
            m.addConstr(s[i] - s[j] + num_customers * sum(x[i, j, k] for k in range(vehicles)) <= num_customers - 1)

# only 1 depot can be built
#m.addConstr(sum(y[i,j] for i in range(num_locations) for j in range(num_capacity)) <= 1)

# m.Params.TimeLimit=60
# m.Params.MIPGap=0.1

m.optimize()


end_time_total = time.process_time()
timer_total = end_time_total - start_time_total

print("Objective: " + str(m.objVal))

#count how many customers are delivered from which depot: use dict of routes, count >0


#for v in m.getVars():
#    print ('%s %g' % ( v . varName , v . x )) #v.varname is x & y; v.x is the binary variable

# display optimal values of decision variables
for depots_coords, depots_capacity in y.keys(): #y: location, capacity
    if (abs(y[depots_coords, depots_capacity].x) > 1e-6):
        print(f"\n Build a warehouse (at location, with capacity type) {depots_coords} {depots_capacity}.")

result_shipping_cost = 0
for distance, distance, vehicles in x.keys(): #x: all location, all location, vehicle
    if (abs(x[distance, distance, vehicles].x) > 1e-6):
        result_shipping_cost = result_shipping_cost + shipping_cost[distance, distance]

result_picking_cost = 0
for customers, depots_coords, depots_capacity in z.keys(): #z: customers, location, capacity
    if (abs(z[customers, depots_coords, depots_capacity].x) > 1e-6):
        result_picking_cost = result_picking_cost + depots_ordercost[depots_coords][depots_capacity]

result_fixed_cost = 0
for depots_coords, depots_capacity in y.keys(): #y: location, capacity
    if (abs(y[depots_coords, depots_capacity].x) > 1e-6):
        result_fixed_cost = result_fixed_cost + depots_fixedCost[depots_coords][depots_capacity] + depots_inboundCost[depots_coords][depots_capacity]