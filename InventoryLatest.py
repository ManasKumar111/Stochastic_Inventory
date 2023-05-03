from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LinearRegression

def optimize_inventory(products, sales_data):
    # Determine number of products
    num_products = len(products)

    # Initialize arrays to hold results
    expected_profits = np.zeros(num_products)
    optimal_qtys = np.zeros(num_products)
    optimal_safety_stocks = np.zeros(num_products)
    predicted_demands=np.zeros(num_products)

    # Train a linear regression model on the sales data
    X = sales_data[:, :-1]
    y = sales_data[:, -1]
    model = LinearRegression()
    model.fit(X, y)

    # Loop over each product
    for i, product in enumerate(products):
        # Extract product details
        product_name = product['product_name']
        purchase_cost = product['purchase_cost']
        selling_price = product['selling_price']
        holding_cost = product['holding_cost']
        shortage_cost = product['shortage_cost']
        mean_lead_time = product['mean_lead_time']
        std_lead_time = product['std_lead_time']
        capacity = product['capacity']
        reorder_point = product['reorder_point']
        safety_stocks = product['safety_stocks']

        # Use the model to predict demand for the next lead time
        X_pred = np.array([[
            product['mean_demand'],
            product['std_demand'],
            product['mean_lead_time'],
            product['std_lead_time']
        ]])
        predicted_demand = model.predict(X_pred)[0]
        predicted_demands[i] = predicted_demand

        # Calculate profit margins for each unit sold
        profit_margin = selling_price - purchase_cost

        # Determine the optimal order quantity for each safety stock level
        optimal_qtys_safety_stock = []
        for safety_stock in safety_stocks:
            # Calculate the reorder point
            reorder_point_with_ss = reorder_point + safety_stock

            # Calculate the expected demand during lead time
            expected_demand_lead_time = predicted_demand * mean_lead_time
            std_dev_demand_lead_time = np.sqrt((mean_lead_time * product['std_demand'])**2 + (predicted_demand * std_lead_time)**2)

            # Calculate the optimal order quantity
            optimal_qty = np.ceil(expected_demand_lead_time - capacity + reorder_point_with_ss)
            if optimal_qty < reorder_point_with_ss:
                optimal_qty = reorder_point_with_ss
            optimal_qtys_safety_stock.append(optimal_qty)

        # Calculate expected profits for each safety stock level
        expected_profits_safety_stock = []
        for j, safety_stock in enumerate(safety_stocks):
            # Calculate the reorder point
            reorder_point_with_ss = reorder_point + safety_stock

            # Calculate the expected demand during lead time
            expected_demand_lead_time = predicted_demand * mean_lead_time
            std_dev_demand_lead_time = np.sqrt((mean_lead_time * product['std_demand'])**2 + (predicted_demand * std_lead_time)**2)

            # Calculate the optimal order quantity
            optimal_qty = optimal_qtys_safety_stock[j]

            # Calculate the probability of stocking out
            z = (reorder_point_with_ss - expected_demand_lead_time) / std_dev_demand_lead_time
            p_stockout = 1 - norm.cdf(z)

            # Calculate the expected profit
            expected_sales = min(expected_demand_lead_time, optimal_qty) * selling_price
            expected_holding_cost = (optimal_qty - expected_demand_lead_time) * holding_cost / 2
            expected_shortage_cost = (expected_demand_lead_time - optimal_qty) * shortage_cost * p_stockout
            expected_profit = expected_sales - purchase_cost * optimal_qty - expected_holding_cost - expected_shortage_cost

            # Store the results
            expected_profits_safety_stock.append(expected_profit)

            # Determine the optimal safety stock level for this product
        optimal_index = np.argmax(expected_profits_safety_stock)
        optimal_safety_stock = safety_stocks[optimal_index]
        optimal_qty = optimal_qtys_safety_stock[optimal_index]
        expected_profit = expected_profits_safety_stock[optimal_index]

        # Store the results for this product
        expected_profits[i] = expected_profit
        optimal_qtys[i] = optimal_qty
        optimal_safety_stocks[i] = optimal_safety_stock


    # Return the optimal results for all products
    return expected_profits,optimal_qtys,optimal_safety_stocks,predicted_demands
#
products = [
    {
        'product_name': 'Amul Buttermilk', 
        'purchase_cost': 12.60,
        'selling_price': 14.00,
        'holding_cost': 0.2175,
        'shortage_cost': 1,
        'mean_demand': 1407.5,
        'std_demand': 427.9363801,
        'mean_lead_time': 5,
        'std_lead_time': 2.5,
        'capacity': 1000,
        'reorder_point': 15,
        'safety_stocks': [16, 16, 27, 38, 40, 4, 4, 16,	16,	14,	10,	9]
    },
    {
        'product_name': 'Amul Lassi',
        'purchase_cost': 15.3,
        'selling_price': 17,
        'holding_cost': 0.2175,
        'shortage_cost': 2,
        'mean_demand': 428.3333333,
        'std_demand': 237.9011763,
        'mean_lead_time': 3,
        'std_lead_time': 1.5,
        'capacity': 450,
        'reorder_point': 15,
        'safety_stocks': [4, 5,	8, 16, 18, 1, 1, 5, 4, 4, 4, 3]
    },
    {
        'product_name': 'Amul Stick Ice-Creams', 
        'purchase_cost': 29.75,
        'selling_price': 35,
        'holding_cost': 0.2175,
        'shortage_cost': 3,
        'mean_demand': 410.8333333,
        'std_demand': 156.9862955,
        'mean_lead_time': 4,
        'std_lead_time': 2,
        'capacity': 300,
        'reorder_point': 15,
        'safety_stocks': [4, 5,	8, 12, 12, 1, 1, 6, 5, 5, 4, 3]
    },
    {
        'product_name': 'Amul Ice-Cream Cones', 
        'purchase_cost': 34,
        'selling_price': 40,
        'holding_cost': 0.2175,
        'shortage_cost': 2,
        'mean_demand': 1175.416667,
        'std_demand': 298.9029816,
        'mean_lead_time': 2,
        'std_lead_time': 1,
        'capacity': 800,
        'reorder_point': 15,
        'safety_stocks': [10, 12, 23, 32, 32, 4, 4, 14, 14, 10,	10,	9]

    },
    {
        'product_name': 'Flavoured Milk', 
        'purchase_cost': 31.5,
        'selling_price': 35,
        'holding_cost': 0.2175,
        'shortage_cost': 1,
        'mean_demand': 3054.166667,
        'std_demand': 345.3972241,
        'mean_lead_time': 5,
        'std_lead_time': 2.5,
        'capacity': 1800,
        'reorder_point': 15,
        'safety_stocks': [28, 30, 48, 68, 72, 13, 13, 32, 33, 33, 30, 30]
    },
    {
        'product_name': 'Cold Drinks(250ml)', 
        'purchase_cost': 18,
        'selling_price': 20,
        'holding_cost': 0.2175,
        'shortage_cost': 2,
        'mean_demand': 539.1666667,
        'std_demand': 164.8943647,
        'mean_lead_time': 3,
        'std_lead_time': 1.5,
        'capacity': 350,
        'reorder_point': 15,
        'safety_stocks': [5, 6,	10, 14,	14,	1, 2, 7, 6, 6, 6, 6]

    },
    {
        'product_name': 'Cold Drinks(500ml)', 
        'purchase_cost': 36,
        'selling_price': 40,
        'holding_cost': 0.2175,
        'shortage_cost': 3,
        'mean_demand': 824.1666667,
        'std_demand': 185.7886548,
        'mean_lead_time': 4,
        'std_lead_time': 2,
        'capacity': 500,
        'reorder_point': 15,
        'safety_stocks': [8, 9, 14, 20, 20,	3, 2, 10, 10, 9, 9, 8]
    },
    {
        'product_name': 'Sting', 
        'purchase_cost': 18,
        'selling_price': 20,
        'holding_cost': 0.2175,
        'shortage_cost': 2,
        'mean_demand': 1745,
        'std_demand': 353.2833012,
        'mean_lead_time': 2,
        'std_lead_time': 1,
        'capacity': 1000,
        'reorder_point': 15,
        'safety_stocks': [18, 19, 29, 38, 40, 5, 5, 18, 19, 19, 19,	20]
    },
    {
        'product_name': 'Kulfi', 
        'purchase_cost': 22.5,
        'selling_price': 25,
        'holding_cost': 0.2175,
        'shortage_cost': 1,
        'mean_demand': 1236.666667,
        'std_demand': 296.4128979,
        'mean_lead_time': 4,
        'std_lead_time': 2,
        'capacity': 800,
        'reorder_point': 15,
        'safety_stocks': [12, 13, 21, 30, 32, 4, 4, 15, 15, 14, 12,	10]
    },
    {
        'product_name': 'PaperBoat', 
        'purchase_cost': 17,
        'selling_price': 20,
        'holding_cost': 0.2175,
        'shortage_cost': 3,
        'mean_demand': 1695.833333,
        'std_demand': 331.9764048,
        'mean_lead_time': 3,
        'std_lead_time': 1.5,
        'capacity': 1000,
        'reorder_point': 15,
        'safety_stocks': [18, 18, 27, 39, 40, 5, 5,	18, 18, 18, 18, 18]
    },
    {
        'product_name': 'Ice-Cream Sandwich', 
        'purchase_cost': 27,
        'selling_price': 30,
        'holding_cost': 0.2175,
        'shortage_cost': 2,
        'mean_demand': 1366.666667,
        'std_demand': 307.7287274,
        'mean_lead_time': 2,
        'std_lead_time': 1,
        'capacity': 533.33,
        'reorder_point': 15,
        'safety_stocks': [14, 15, 24, 32, 32, 4, 4, 16, 15, 15, 15,	14]
    },
    {
        'product_name': 'Chocolates', 
        'purchase_cost': 36,
        'selling_price': 40,
        'holding_cost': 0.2175,
        'shortage_cost': 1,
        'mean_demand': 1383.75,
        'std_demand': 279.5542312,
        'mean_lead_time': 5,
        'std_lead_time': 2.5,
        'capacity': 533.33,
        'reorder_point': 15,
        'safety_stocks': [14, 14, 23, 32, 32, 4, 4,	16,	15,	15,	16,	15]
    },
    
]

sales_data = np.array([
    [1407.5, 427.9363801, 5, 2.5, 1000],
    [428.3333333, 237.9011763, 3, 1.5, 450],
    [410.8333333, 156.9862955, 4, 2, 300],
    [1175.416667, 298.9029816, 2, 1, 800],
    [3054.166667, 345.3972241, 5, 2.5, 1800],
    [539.1666667, 164.8943647, 3, 1.5, 350],
    [824.1666667, 185.7886548, 4, 2, 500],
    [1745, 353.2833012, 2, 1, 1000],
    [1236.666667, 296.4128979, 4, 2, 800],
    [1695.833333,331.9764048, 3, 1.5, 1000],
    [1366.666667, 307.7287274, 2, 1, 533.3333333],
    [1383.75, 279.5542312, 5, 2.5, 533.3333333]
]) 



# Call the function to optimize inventory levels
expected_profits, optimal_qtys, optimal_safety_stocks,predicted_demands = optimize_inventory(products, sales_data)

EXPECTED_profits= np.sum(expected_profits)


print(EXPECTED_profits)
print(optimal_safety_stocks)
print(optimal_qtys)
print(predicted_demands)