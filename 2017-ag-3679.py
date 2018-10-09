from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as dd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm



b= dd.DataFrame({"Fertilizer": [100,200,300,400,500,600,700], "Rainfall": [10, 20, 10, 30, 20, 20, 30], "Yield": [40, 50, 50, 70, 65, 65, 80]})
result = smf.ols(formula="Yield ~ Fertilizer + Rainfall", data=b).fit()
print(result.summary())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(b['Fertilizer'],
           b['Rainfall'], b['Yield'],
           c='r', marker='o')
ab, ba = np.meshgrid(b['Fertilizer'],
                     b['Rainfall'])
exog = dd.core.frame.DataFrame({'Fertilizer':ab.ravel(),
                                'Rainfall':ba.ravel()}
                               )
out = result.predict(exog=exog)
ax.plot_surface(ab, ba, out.values.reshape(ab.shape), rstride=1, cstride=1, alpha='0.4', color='red')
ax.set_xlabel("Fertilizer")
ax.set_ylabel("Rainfall")
ax.set_zlabel("Yield")
plt.show()