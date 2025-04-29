import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('AI\ML\data.csv')

plt.scatter(data.x,data.y)
plt.show()