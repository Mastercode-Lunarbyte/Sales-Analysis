#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# خواندن داده‌های فروش
file_path = "sales_data.csv"  #، مسیر فایل CSV،
sales_data = pd.read_csv(file_path, parse_dates=["date"])

# پردازش داده‌ها
sales_data["month"] = sales_data["date"].dt.to_period("M")
monthly_sales = sales_data.groupby("month")["revenue"].sum().reset_index()
monthly_sales["month"] = monthly_sales["month"].astype(str)

# رسم نمودار روند درآمد ماهانه
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_sales, x="month", y="revenue", marker="o", label="Revenue")
plt.xticks(rotation=45)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.show()

# پیش‌بینی درآمد ماه‌های آینده
X = np.arange(len(monthly_sales)).reshape(-1, 1)
y = monthly_sales["revenue"].values

model = LinearRegression()
model.fit(X, y)

# پیش‌بینی 3 ماه آینده
future_months = np.arange(len(monthly_sales), len(monthly_sales) + 3).reshape(-1, 1)
predicted_revenue = model.predict(future_months)

# نمایش پیش‌بینی
for i, rev in enumerate(predicted_revenue, 1):
    print(f"Predicted Revenue for Month {i}: {rev:.2f}")


# In[ ]:




