import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv(r"D:/datasets/used_cars.csv")


df["price"] = df["price"].str.replace("$","",regex=False)
df["price"] = df["price"].str.replace(",","",regex=False)
df["price"] =df["price"].astype(int)
df = df[df["price"]<300000]

df["milage"] = df["milage"].str.replace("mi.","",regex=False)
df["milage"] = df["milage"].str.replace(",","",regex=False)
df["milage"] = df["milage"].astype(int)

df["fuel_type"] = df["fuel_type"].fillna("Unknown")
print(df["fuel_type"].value_counts(dropna=False))
df["accident"] = df["accident"].fillna("Unknown")
print(df["accident"].value_counts(dropna=False))
df.drop(columns=["clean_title"], inplace=True)
df["Car_Age"] = 2026 -df["model_year"]


top10 = df["brand"].value_counts().head(10).index
top_pr = df[df["brand"].isin(top10)]
plt.figure(figsize=(12,6))
sns.boxplot(x = "brand",y= "price",data = top_pr)
plt.title("brand vs price")
plt.xticks(rotation = 45)
plt.show()
plt.figure(figsize=(10,5))
plt.bar(top10, top10.values)
plt.xticks(rotation=45)
plt.title("Top 10 Cars Brands")
plt.xlabel("Brand")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(df["milage"],df["price"])
plt.title("Milage vs Price")
plt.xlabel("Milage")
plt.ylabel("Price")
plt.show()
year_price = df.groupby("model_year")["price"].mean().sort_index()
plt.figure(figsize=(10,5))
plt.plot(year_price.index, year_price.values)
plt.title("Average Price by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Average Price")
plt.show()
df.groupby("fuel_type")["price"].mean().plot(figsize=(10,5))
plt.title("Average Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Price")
plt.show()
df.groupby("accident")["price"].mean().plot(figsize=(10,5))
plt.title("Average Price by Accident Status")
plt.xlabel("Accident Status")
plt.ylabel("Average Price")
plt.show()
print(df["milage"].corr(df["price"]))


top_models =df.groupby("model")["price"].mean().sort_values(ascending=False).head(10)
print(top_models)
df["price"] = np.log1p(df["price"])
x = df.drop("price",axis =1)
cat_cols = x.select_dtypes(include=["object","string","category"]).columns.tolist()
for col in cat_cols:
    cat = pd.Categorical(x[col]).categories
    x[col] = pd.Categorical(x[col],categories=cat) 
y = df["price"]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error 
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from xgboost import XGBRegressor
rf_model = XGBRegressor(n_estimators=300,learning_rate = 0.05,subsample = 0.8,colsample_bytree = 0.8,
                        tree_method="hist",enable_categorical = True,max_depth = 6,random_state=42)
rf_model.fit(x_train,y_train)
rf_pred = (rf_model.predict(x_test))
rf_pred_final = np.expm1(rf_pred)
y_test_final = np.expm1(y_test)
print(root_mean_squared_error(y_test_final,rf_pred_final))
print(r2_score(y_test_final,rf_pred_final))
print("mse Absolute Error:", mean_absolute_error(y_test_final, rf_pred_final))
results1 = x_test.copy()
results1["Actual Price"] = y_test_final
results1["Predicted Price"] = rf_pred_final
print(results1.head(10))


