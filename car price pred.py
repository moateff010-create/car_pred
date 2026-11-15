import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv(r"D:/datasets/used_cars.csv")
# -- duplication check --
duplicated_count = df.duplicated().sum()
print(f"number of duplications: {duplicated_count}")
if duplicated_count>0:
    df = df.drop_duplicates()

df["price"] = df["price"].str.replace("$","",regex=False)
df["price"] = df["price"].str.replace(",","",regex=False)
df["price"] =df["price"].astype(int)
df = df[df["price"]<300000]

print(df["milage"].describe())
df["milage"] = df["milage"].str.replace("mi.","",regex=False)
df["milage"] = df["milage"].str.replace(",","",regex=False)
df["milage"] = df["milage"].astype(int)

# --- IQR outlier method ---
Q1 = df["milage"].quantile(0.25)
Q3 = df["milage"].quantile(0.75)
IQR = Q3-Q1
lower_bond = Q1 - 1.5 * IQR
upper_bond = Q3 + 1.5 * IQR
df["milage"] = df["milage"].clip(lower=lower_bond,upper=upper_bond)

print(df.isnull().sum())
df["fuel_type"] = df["fuel_type"].fillna("Unknown")
print(df["fuel_type"].value_counts(dropna=False))
df["accident"] = df["accident"].fillna("Unknown")
print(df["accident"].value_counts(dropna=False))
df.drop(columns=["clean_title"], inplace=True)
print(df.isnull().sum())
df["Car_Age"] = 2026 -df["model_year"]



top10 = df["brand"].value_counts().head(10)
plt.figure(figsize=(12,6))
sns.barplot(x = top10.index,y= top10.values,palette="coolwarm")
plt.title("brand vs price")
plt.xlabel("brand")
plt.ylabel("price")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.bar(top10.index, top10.values)
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

df.groupby("fuel_type")["price"].mean().plot(kind="bar",figsize=(10,5))
plt.title("Average Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Price")
plt.show()

df.groupby("accident")["price"].mean().plot(kind="bar",figsize=(10,5))
plt.title("Average Price by Accident Status")
plt.xlabel("Accident Status")
plt.ylabel("Average Price")
plt.show()

print(df["milage"].corr(df["price"]))


top_models =df.groupby("model")["price"].mean().sort_values(ascending=False).head(10)
print(top_models)

# -- skewness --
skew_before = df["price"].skew()
print(f"skewness before log: {skew_before:.2f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df["price"],kde = True,color = "red")
plt.title(f"price distribution before log: {skew_before:.2f}")
plt.xlabel("price")
plt.ylabel("count")

df["price"] = np.log1p(df["price"])
skew_after = df["price"].skew()
print(f"skewness after log {skew_after:.2f}")

plt.subplot(1,2,2)
sns.histplot(df["price"],kde = True,color = "green")
plt.title(f"price distribution after log{skew_after:.2f}")
plt.xlabel("log(price + 1)")
plt.ylabel("count")
plt.tight_layout()
plt.show()

x = df.drop("price",axis =1)
cat_cols = x.select_dtypes(include=["object","string","category"]).columns.tolist()
for col in cat_cols:
    cat = pd.Categorical(x[col]).categories
    x[col] = pd.Categorical(x[col],categories=cat) 
y = df["price"]
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error 
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -- modeling -- 
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

# --- rmse = 22812.618202258007
# --- r2 score = 0.6563839468445258

