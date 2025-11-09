import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('US_births_2000-2014_SSA.csv')

data['date'] = pd.to_datetime({
    'year':  data['year'],
    'month': data['month'],
    'day': data['date_of_month']
})
data['total_date'] = (data['date'] - data['date'].min()).dt.days

print(data.head())

plt.plot(data['total_date'], data['births'])
plt.show()