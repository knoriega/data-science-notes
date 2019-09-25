# DataCamp: Intro to Risk Mgmt in Python Course

## Snippets

*Skewness and Excess Kurtosis*
```
from scipy.stats import skew
from scipy.stats import kurtosis
```

*Shapiro-Wilk Test for Normality: H0 -- Data is normal*
```
from scipy.stats import shapiro
```

*Cumulative Simple Return*
```
# Plot the cumulative portfolio returns over time
CumulativeReturns = ((1+StockReturns["Portfolio"]).cumprod()-1)
```

*Correlation Matrix*
```
# Import seaborn as sns
import seaborn as sns

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()
```

*Another linear regression  -- CAPM*
```
# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
FamaFrench5_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML + RMW + CMA ', data=FamaFrenchData)

# Fit the regression
FamaFrench5_fit = FamaFrench5_model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = FamaFrench5_fit.rsquared_adj
print(regression_adj_rsq)
```

*CVaR Plotting*
```
# Historical CVaR 95
cvar_95 = StockReturns_perc[StockReturns_perc < var_95].mean()
print(cvar_95)

# Sort the returns for plotting
sorted_rets = sorted(StockReturns_perc)

# Plot the probability of each return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 and CVaR 95 quantiles
plt.axvline(x=var_95, color="r", linestyle="-", label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(x=cvar_95, color='b', linestyle='-', label='CVaR 95: {0:.2f}%'.format(cvar_95))
plt.show()
```

*Using Normal Dist for Var*
```
# Import norm from scipy.stats
from scipy.stats import norm

# Estimate the average daily return
mu = np.mean(StockReturns)

# Estimate the daily volatility
vol = np.std(StockReturns)

# Set the VaR confidence level
confidence_level = 0.05

# Calculate Parametric VaR
var_95 = norm.ppf(confidence_level, mu, vol)
print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95))
```

*MC Random Walks vectorized (sans the list comprehension)*
```
# Using matricies
mc = np.empty((100, 252))
rand_rets = np.random.normal(mu, vol, 100 * 252).reshape(100, 252)
paths = S0*(rand_rets+1).cumprod(axis=1)
[plt.plot(range(T), paths[i, :]) for i in range(np.shape(rand_rets)[0])]
plt.show()
```
