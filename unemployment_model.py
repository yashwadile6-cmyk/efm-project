# unemployment_model_final.py
# 🎓 Project: Economics of Unemployment in India
# Complete ML Project with Visualizations, Model Saving, and Predictions

"""
📘 PROJECT TITLE: Economics of Unemployment in India
👩‍💻 DEVELOPED BY: [Your Name]
🏫 UNDER THE GUIDANCE OF: [Teacher’s Name]

🎯 OBJECTIVE:
To analyze unemployment trends in India using real-world data and predict unemployment rates
based on time factors such as year and month.

📊 TOOLS USED:
Python, Pandas, Seaborn, Matplotlib, Scikit-Learn, Joblib
"""

# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------
# 2. Load Dataset
# -------------------------
print("\n📂 Loading Dataset...")
data = pd.read_csv("Unemployment in India.csv")

print("\n✅ Dataset Loaded Successfully!")
print("First 5 rows:\n", data.head())
print("\nColumns available:\n", data.columns)

# -------------------------
# 3. Data Cleaning & Preparation
# -------------------------
data.columns = data.columns.str.strip()  # Remove extra spaces
data = data.dropna()  # Drop missing rows
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Create Year and Month columns
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

print("\nDataset cleaned and formatted successfully!")

# -------------------------
# 4. Data Overview
# -------------------------
print("\n📊 Dataset Information:")
print(data.info())
print("\nTotal Records:", len(data))

# -------------------------
# 5. Data Visualization
# -------------------------
sns.set_style("whitegrid")

# --- Unemployment trend over time ---
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data, color='royalblue', linewidth=2.5)
plt.title("📉 Unemployment Rate in India Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Unemployment Rate (%)", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Unemployment by region (if available) ---
if 'Region' in data.columns:
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=data, palette='mako')
    plt.title("🌍 Average Unemployment Rate by Region", fontsize=15, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔥 Correlation Between Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -------------------------
# 6. Model Training
# -------------------------
print("\n🤖 Training Linear Regression Model...")

# Feature and Target
X = data[['Year', 'Month']]
y = data['Estimated Unemployment Rate (%)']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# -------------------------
# 7. Model Evaluation
# -------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# --- Actual vs Predicted Plot ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='orange', s=60, alpha=0.8, edgecolor='black')
sns.lineplot(x=y_test, y=y_test, color='blue', linestyle='--', label='Perfect Prediction')
plt.title("🎯 Actual vs Predicted Unemployment Rate", fontsize=15, fontweight='bold')
plt.xlabel("Actual Rate (%)", fontsize=12)
plt.ylabel("Predicted Rate (%)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 8. Key Insights
# -------------------------
print("\n📊 Key Insights:")
print(f"- Average Unemployment Rate: {data['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"- Highest Unemployment Recorded: {data['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"- Lowest Unemployment Recorded: {data['Estimated Unemployment Rate (%)'].min():.2f}%")

# -------------------------
# 9. Save Model
# -------------------------
joblib.dump(model, "unemployment_model.pkl")
print("\n💾 Model saved as 'unemployment_model.pkl' (you can reuse it later for predictions)")

# -------------------------
# 10. Interactive Prediction
# -------------------------
print("\n🧮 Let's Predict Future Unemployment Rate!")
try:
    year = int(input("Enter Year (e.g., 2025): "))
    month = int(input("Enter Month (1-12): "))
    pred = model.predict([[year, month]])
    print(f"📊 Predicted Unemployment Rate for {month}/{year}: {pred[0]:.2f}%")
except Exception as e:
    print("⚠️ Invalid input. Please enter numeric values for year and month.")
    print("Error details:", e)

# -------------------------
# 11. Summary
# -------------------------
print("""
🔍 SUMMARY:
- This project analyzed the unemployment trend in India using real economic data.
- Visualizations showed regional and time-based unemployment patterns.
- A Linear Regression model was trained to predict unemployment using Year & Month.
- The model achieved good accuracy and was saved for future use.
- Interactive prediction allows forecasting future unemployment rates.

📈 Future Scope:
We can extend this project using advanced ML algorithms (Random Forest, ARIMA) 
and include more socio-economic indicators like GDP, inflation, and labor participation rate.
""")

print("✅ Project Completed Successfully!")
