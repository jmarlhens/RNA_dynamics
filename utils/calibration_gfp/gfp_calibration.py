import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# Load the provided file to inspect the data structure
file_path = "gfp_Calibration.csv"
data = pd.read_csv(file_path)

# Display the first few rows to understand its structure
data.head()

# Prepare the data
gfp_concentration = data["GFP Concentration (nM)"]
fluorescence_replicates = data.filter(like="F.I. (a.u)")

# Calculate mean and confidence intervals
mean_fluorescence = fluorescence_replicates.mean(axis=1)
std_fluorescence = fluorescence_replicates.std(axis=1)
confidence_interval = 1.96 * (
    std_fluorescence / np.sqrt(fluorescence_replicates.shape[1])
)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot individual replicates
for replicate in fluorescence_replicates.columns:
    plt.scatter(
        gfp_concentration,
        fluorescence_replicates[replicate],
        alpha=0.5,
        label=f"Replicate: {replicate}",
    )

# Plot mean and confidence intervals
plt.errorbar(
    gfp_concentration,
    mean_fluorescence,
    yerr=confidence_interval,
    fmt="-o",
    color="black",
    label="Mean ± 95% CI",
    capsize=5,
)

# Formatting the plot
plt.title("GFP Calibration Curve")
plt.xlabel("GFP Concentration (nM)")
plt.ylabel("Fluorescence Intensity (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Perform linear regression on the mean fluorescence data
slope, intercept, r_value, p_value, std_err = linregress(
    gfp_concentration, mean_fluorescence
)

# Create a linear fit line
fit_line = slope * gfp_concentration + intercept

# Plot the data with the regression line
plt.figure(figsize=(10, 6))

# Scatter plot of mean fluorescence
plt.errorbar(
    gfp_concentration,
    mean_fluorescence,
    yerr=confidence_interval,
    fmt="o",
    label="Mean ± 95% CI",
    color="blue",
    capsize=5,
)

# Regression line
plt.plot(
    gfp_concentration,
    fit_line,
    color="red",
    label=f"Linear Fit: y = {slope:.2f}x + {intercept:.2f}",
)

# Formatting the plot
plt.title("GFP Calibration Curve with Linear Regression")
plt.xlabel("GFP Concentration (nM)")
plt.ylabel("Fluorescence Intensity (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display regression parameters
regression_params = {
    "Slope": slope,
    "Intercept": intercept,
    "R-squared": r_value**2,
    "P-value": p_value,
    "Std. Error": std_err,
}

# Show the plot and parameters
plt.show()
regression_params
#
#
# from statsmodels.stats.diagnostic import het_breuschpagan
# import statsmodels.api as sm
#
# # Calculate residuals
# predicted = slope * gfp_concentration + intercept
# residuals = mean_fluorescence - predicted
#
# # Plot residuals
# plt.figure(figsize=(10, 6))
# plt.scatter(gfp_concentration, residuals, alpha=0.7, color='blue', label='Residuals')
# p

# Calculate the variance for each concentration
variance_fluorescence = fluorescence_replicates.var(axis=1)


# Define the linear model with zero intercept
def linear_model(concentration, factor):
    return factor * concentration


# Fit the model to variance vs concentration
params, covariance = curve_fit(linear_model, gfp_concentration, variance_fluorescence)

# Extract the factor (slope)
factor = params[0]
print(f"Estimated factor: {factor:.4f}")

# Predict variance using the fitted model
predicted_variance = factor * gfp_concentration

# Plot observed vs predicted variance
plt.figure(figsize=(10, 6))
plt.scatter(
    gfp_concentration, variance_fluorescence, label="Observed Variance", color="blue"
)
plt.plot(
    gfp_concentration,
    predicted_variance,
    label=f"Predicted Variance: factor={factor:.4f}",
    color="red",
)
plt.title("Observed vs. Predicted Variance")
plt.xlabel("GFP Concentration (nM)")
plt.ylabel("Variance of Fluorescence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals
residuals = variance_fluorescence - predicted_variance

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(gfp_concentration, residuals, label="Residuals", color="purple")
plt.axhline(0, color="black", linestyle="--")
plt.title("Residuals of Variance Model")
plt.xlabel("GFP Concentration (nM)")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate R-squared
ss_total = np.sum((variance_fluorescence - np.mean(variance_fluorescence)) ** 2)
ss_residual = np.sum(residuals**2)
r_squared = 1 - (ss_residual / ss_total)
print(f"R-squared: {r_squared:.4f}")
