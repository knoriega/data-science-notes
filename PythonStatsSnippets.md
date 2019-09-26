# DataCamp: Statistics Interview Question Prep

## scipy
*Confidence Interval*
```
import scipy.stats as st
a = range(10,14)
st.t.interval(0.95, len(a) - 1, loc = np.mean(a),
scale = st.sem(a))
```

*Two-tailed test for identical means*
```
from scipy.stats import ttest_ind
tstat, pval = ttest_ind(asus, toshiba)
print('{0:0.3f}'.format(pval))
```

## statsmodels
*Proportion Confidence Interval*
```
from sm.stats.proportional import proportion_conf
proportion_conf(successes, #trials size, alpha)
```

*Proportion One-Tailed T-test*
```
from statsmodels.stats.proportion import proportions_ztest
count = np.array([num_treat, num_control]) 
nobs = np.array([total_treat, total_control])

# Run the z-test and print the result 
stat, pval = proportions_ztest(count, nobs, alternative="larger")
print('{0:0.3f}'.format(pval))
```

*Standardize effect size (i.e desired before and after change)*
```
# Standardize the effect size
from statsmodels.stats.proportion import proportion_effectsize
std_effect = proportion_effectsize(.20, .25)
```

*Find desired sample size based on alpha, power, and effect size*
```
# Assign and print the needed sample size
from statsmodels.stats.power import  zt_ind_solve_power
sample_size = zt_ind_solve_power(effect_size=std_effect, nobs1=None, alpha=0.05, power=0.95)
print(sample_size)
```

*Viz: Change in power based on number of obs and effect size*
```
sample_sizes = np.array(range(5, 100))
effect_sizes = np.array([0.2, 0.5, 0.8])

# Create results object for t-test analysis
from statsmodels.stats.power import TTestIndPower
results = TTestIndPower()

# Plot the power analysis
results.plot_power(dep_var='nobs', nobs=sample_sizes, effect_size=effect_sizes)
plt.show()
```

*Multiple T-test pval correction*
```
from statsmodels.sandbox.stats.multicomp import multipletests
pvals = [.01, .05, .10, .50, .99]

# Create a list of the adjusted p-values
p_adjusted = multipletests(pvals, alpha=0.05, method='bonferroni')

# Print the resulting conclusions
print(p_adjusted[0])

# Print the adjusted p-values themselves 
print(p_adjusted[1])
```

## scikitlearn

*Linear Regression -- Basic workflow*
```
from sklearn.linear_model import LinearRegression 
X_train = np.array(weather['Humidity9am']).reshape(-1,1)
y_train = weather['Humidity3pm']

# Create and fit your linear regression model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Assign and print predictions
preds = lm.predict(X_train)

# Assign and print coefficient 
coef = lm.coef_
print(coef)
```

*MSE or MAE (just sub-in the type needed)*
```
# Mean squared error
from sklearn.metrics import mean_squared_error
preds = lm.predict(X)
mse = mean_squared_error(preds, y)
print(mse)
```

*Confusion matricies*
```
# Generate and output the confusion matrix
from sklearn.metrics import confusion_matrix
preds = clf.predict(X_test)
matrix = confusion_matrix(y_test, preds)
print(matrix)
```

*Precision and Recall scores*
```
# Compute and print the recall
from sklearn.metrics import recall_score
preds = clf.predict(X_test)
recall = recall_score(y_test, preds)
print(recall)
```
