# Import required libraries
# numpy and pandas for data handling; statsmodels for regression;
# matplotlib for visualization.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------
# 1. Construct the monthly dataset (2021-01 to 2023-12)
# -----------------------------
data = {
    'Month': pd.date_range("2021-01", periods=36, freq='M'),
    'Passenger': [6213.1, 4943.3, 9850.2, 10524.8, 10508.4, 8516.1,
                  10138.6, 4631.5, 7424.4, 7978.0, 4420.8, 5590.3,
                  6105.3, 6471.4, 3171.4, 1625.0, 2501.2, 4555.8,
                  7046.2, 6658.3, 4133.9, 3274.6, 2587.4, 3869.5,
                  8183.9, 8894.6, 9373.8, 10279.1, 10549.7, 10803.3,
                  12670.4, 12944.4, 10833.8, 11353.5, 9901.0, 10163.1],
    'Flights': [75.0, 52.2, 96.4, 99.2, 99.2, 80.6,
                95.2, 66.9, 83.6, 86.9, 64.4, 72.8,
                71.7, 69.4, 53.3, 37.7, 49.8, 64.0,
                84.1, 79.5, 60.1, 47.2, 43.4, 51.5,
                72.9, 85.6, 98.1, 97.9, 103.5, 100.4,
                110.1, 111.0, 101.3, 101.1, 93.9, 94.1],
    'Util': [6.3, 5.3, 8.2, 8.6, 8.2, 7.0,
             7.7, 4.6, 6.7, 6.7, 4.7, 5.6,
             6.2, 6.6, 3.5, 2.2, 3.0, 4.7,
             6.3, 5.8, 3.9, 3.2, 2.8, 3.9,
             6.7, 7.6, 7.5, 8.2, 8.2, 8.3,
             9.1, 9.2, 8.4, 8.2, 7.9, 8.0],
    'Cargo': [158.3, 111.1, 154.8, 157.0, 159.7, 155.3,
              147.1, 133.6, 149.1, 150.2, 149.1, 156.9,
              154.6, 103.3, 129.0, 102.2, 116.2, 128.1,
              128.4, 122.1, 123.3, 116.1, 113.7, 114.9,
              113.6, 105.4, 130.8, 126.7, 133.9, 146.2,
              138.0, 144.3, 157.9, 152.0, 164.3, 164.9]
}

# Put the data into a pandas DataFrame
df = pd.DataFrame(data)

# -----------------------------
# 2. Build the multiple linear regression model
#    Dependent variable: Passenger
#    Independent variables: Flights, Util, Cargo
# -----------------------------
X = df[['Flights', 'Util', 'Cargo']]  # design matrix without intercept
y = df['Passenger']                   # response vector

# Add constant term (intercept) to X
X = sm.add_constant(X)

# Fit the OLS regression model
model = sm.OLS(y, X).fit()

# Print full regression summary:
# includes coefficients, standard errors, t-stats, p-values, R-squared, etc.
print(model.summary())

# Also extract predicted values for later plotting
y_pred = model.predict(X)

# -----------------------------
# 3. Visualization 1:
#    Actual vs. Predicted Passenger Throughput
# -----------------------------
plt.figure()
plt.scatter(df['Passenger'], y_pred, label='Predicted vs Actual')

# Plot a 45-degree reference line
min_val = min(df['Passenger'].min(), y_pred.min())
max_val = max(df['Passenger'].max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val],
         linestyle='--', label='Perfect fit line')

plt.xlabel("Actual Passenger Throughput (10k persons)")
plt.ylabel("Predicted Passenger Throughput (10k persons)")
plt.title("Actual vs Predicted Passenger Throughput")
plt.legend()
plt.tight_layout()

# Save figure to file (used in LaTeX as fig_actual_vs_pred.png)
plt.savefig("fig_actual_vs_pred.png", dpi=300)
plt.close()

# -----------------------------
# 4. Visualization 2:
#    Flights vs. Passenger Throughput (with simple linear fit)
# -----------------------------
plt.figure()
plt.scatter(df['Flights'], df['Passenger'], label='Observed points')

# Fit a simple linear trend line for visualization:
coef = np.polyfit(df['Flights'], df['Passenger'], 1)
x_line = np.linspace(df['Flights'].min(), df['Flights'].max(), 100)
y_line = np.polyval(coef, x_line)
plt.plot(x_line, y_line,
         linestyle='--', label='Linear fit')

plt.xlabel("Flights (10k flights)")
plt.ylabel("Passenger Throughput (10k persons)")
plt.title("Flights vs Passenger Throughput")
plt.legend()
plt.tight_layout()

# Save figure to file (used in LaTeX as fig_flights_vs_passenger.png)
plt.savefig("fig_flights_vs_passenger.png", dpi=300)
