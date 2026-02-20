# =============================================================
# 🏠 House Price Prediction — Using scikit-learn Library
# =============================================================
# Is file mein hum WAHI kaam karenge jo scratch wali file mein
# kiya tha, lekin ab scikit-learn library use karenge.
#
# Faida: Sirf 3-4 lines mein poora model ban jata hai!
# Lekin tumne pehle scratch se seekh liya, toh ab tum samajhte
# ho ke ANDAR kya ho raha hai. 🧠
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Data split karne ke liye
from sklearn.linear_model import LinearRegression      # ML model
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation metrics

# =============================================================
# 📌 STEP 1: Data Load Karo
# =============================================================
print("=" * 60)
print("📌 STEP 1: Data Loading")
print("=" * 60)

data = pd.read_csv("data/house_prices.csv")
print(f"Total rows: {len(data)}")
print(data.head())

# =============================================================
# 📌 STEP 2: Features aur Target Alag Karo
# =============================================================
# Features (X) = jo cheezein model ko INPUT mein doge
# Target  (y) = jo cheez model ko PREDICT karni hai

print("\n" + "=" * 60)
print("📌 STEP 2: Feature Selection")
print("=" * 60)

# Is baar HUM SAB features use karenge (Multiple Linear Regression!)
# Scratch mein sirf Area use kiya tha, ab Area + Bedrooms + Age
X = data[['Area', 'Bedrooms', 'Age']]  # Multiple features
y = data['Price']                        # Target

print(f"Features (X): {list(X.columns)}")
print(f"Target (y): Price")
print(f"X shape: {X.shape}")  # (rows, columns)
print(f"y shape: {y.shape}")  # (rows,)

# =============================================================
# 📌 STEP 3: Train/Test Split
# =============================================================
# sklearn mein yeh SIRF 1 line mein ho jata hai!
# test_size=0.2 = 20% test ke liye, 80% training ke liye
# random_state=42 = har baar same split mile (reproducibility)

print("\n" + "=" * 60)
print("📌 STEP 3: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# =============================================================
# 📌 STEP 4: Model Banao aur Train Karo
# =============================================================
# YAHAN dekhlo — sirf 2 LINES mein model ban gaya aur train bhi
# ho gaya! Lekin yaad rakho, ANDAR mein wahi gradient descent
# ya normal equation chal raha hai jo tumne scratch mein kiya.

print("\n" + "=" * 60)
print("📌 STEP 4: Model Training (scikit-learn)")
print("=" * 60)

model = LinearRegression()   # Model object banao
model.fit(X_train, y_train)  # Training! (yeh 1 line mein sab seekh leta hai)

print("✅ Model trained successfully!")

# Model ne jo coefficients seekhe:
print(f"\n📊 Learned Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   {feature}: {coef:,.2f}")
    # Matlab: har 1 unit increase se price mein kitna fark aata hai

print(f"   Intercept (b): {model.intercept_:,.2f}")
# Intercept = jab sab features 0 hon toh predicted price

# =============================================================
# 📌 STEP 5: Predictions aur Evaluation
# =============================================================
print("\n" + "=" * 60)
print("📌 STEP 5: Model Evaluation")
print("=" * 60)

# Test data pe predict karo
y_pred = model.predict(X_test)

# Metrics calculate karo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Test Results (scikit-learn):")
print(f"   Mean Squared Error (MSE):  {mse:,.0f}")
print(f"   Root MSE (RMSE):           ${rmse:,.0f}")
print(f"   R² Score:                   {r2:.4f}")

# --- Actual vs Predicted Table ---
print(f"\n📋 Actual vs Predicted Prices:")
print(f"   {'Actual':>10} | {'Predicted':>10} | {'Difference':>10}")
print(f"   {'-'*10} | {'-'*10} | {'-'*10}")
y_test_arr = y_test.values
for i in range(len(y_test_arr)):
    diff = y_test_arr[i] - y_pred[i]
    print(f"   ${y_test_arr[i]:>9,.0f} | ${y_pred[i]:>9,.0f} | ${diff:>+9,.0f}")

# =============================================================
# 📌 STEP 6: Visualization
# =============================================================
print("\n" + "=" * 60)
print("📌 STEP 6: Result Visualization")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Plot 1: Predicted vs Actual ---
axes[0].scatter(y_test, y_pred, color='blue', s=80, edgecolors='black')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title('Predicted vs Actual')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Feature Importance (Coefficients) ---
colors = ['#3498db', '#2ecc71', '#e74c3c']
axes[1].barh(X.columns, model.coef_, color=colors, edgecolor='black')
axes[1].set_xlabel('Coefficient Value')
axes[1].set_title('Feature Importance')
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Residuals (Errors) ---
residuals = y_test.values - y_pred
axes[2].bar(range(len(residuals)), residuals, color='orange', edgecolor='black')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Test Sample #')
axes[2].set_ylabel('Error (Actual - Predicted)')
axes[2].set_title('Prediction Errors (Residuals)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step5_sklearn_results.png", dpi=100)
print("✅ Results plot saved: step5_sklearn_results.png")
plt.show()

# =============================================================
# 📌 STEP 7: New Price Predict Karo!
# =============================================================
# Ab apna model use karo — naya ghar ka price predict karo!

print("\n" + "=" * 60)
print("📌 BONUS: Naye Ghar ka Price Predict Karo!")
print("=" * 60)

# Example: 2500 sq ft, 3 bedrooms, 10 years old
new_house = pd.DataFrame({
    'Area': [2500],
    'Bedrooms': [3],
    'Age': [10]
})

predicted_price = model.predict(new_house)
print(f"\n🏠 Naya Ghar:")
print(f"   Area:     2500 sq ft")
print(f"   Bedrooms: 3")
print(f"   Age:      10 years")
print(f"\n💰 Predicted Price: ${predicted_price[0]:,.0f}")

# =============================================================
# 🎉 COMPARISON: Scratch vs sklearn
# =============================================================
print("\n" + "=" * 60)
print("🎉 FINAL SUMMARY: Scratch vs scikit-learn")
print("=" * 60)
print(f"""
┌─────────────────────┬────────────────────┬─────────────────────┐
│ Feature             │ From Scratch       │ scikit-learn         │
├─────────────────────┼────────────────────┼─────────────────────┤
│ Lines of Code       │ ~50 lines          │ ~5 lines             │
│ Features Used       │ Area only          │ Area+Bedrooms+Age    │
│ R² Score            │ (check scratch)    │ {r2:.4f}              │
│ Understanding       │ Deep (math)        │ Surface (API)        │
│ Speed to Build      │ Slow               │ Very Fast            │
│ Best For            │ Learning           │ Production           │
└─────────────────────┴────────────────────┴─────────────────────┘

🧠 KEY LESSON:
   Scratch se seekha → ab tum samajhte ho ML KAISE kaam karta hai
   sklearn se build kiya → ab tum FAST apply kar sakte ho

   DONO zaroori hain! Pehle seekho, phir tools use karo. 🚀
""")
