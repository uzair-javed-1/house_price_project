# =============================================================
# 🏠 House Price Prediction — Linear Regression FROM SCRATCH
# =============================================================
# Yeh file tumhein sikhayegi ke ML model ANDAR se kaise kaam
# karta hai — bina kisi library ke, sirf math aur Python!
#
# STEP 1: Data load karo aur samjho
# STEP 2: Graphs banao (visualization)
# STEP 3: Apna khud ka Linear Regression likho (Gradient Descent)
# STEP 4: Model ko test karo aur results dekho
# =============================================================

import numpy as np        # Math ke liye (arrays, calculations)
import pandas as pd       # CSV file padhne ke liye
import matplotlib.pyplot as plt   # Graphs banane ke liye

# =============================================================
# 📌 STEP 1: Data Load Karo aur Samjho
# =============================================================
# Sabse pehle hum CSV file padhte hain. CSV ka matlab hai
# "Comma Separated Values" — ek simple table format.

print("=" * 60)
print("📌 STEP 1: Data Loading & Exploration")
print("=" * 60)

# CSV file padho
data = pd.read_csv("data/house_prices.csv")

# Pehli 5 rows dikhao — taa ke pata chale data kaisa dikhta hai
print("\n🔍 Pehli 5 rows (data ka preview):")
print(data.head())

# Data ki summary — kitni rows hain, columns kya hain, types kya hain
print("\n📋 Data ki Information:")
print(data.info())

# Statistics — mean, min, max, etc.
print("\n📊 Data ka Summary (Statistics):")
print(data.describe())

# Check karo ke koi missing value toh nahi hai
print("\n❓ Missing values har column mein:")
print(data.isnull().sum())

# =============================================================
# 📌 STEP 2: Data Visualization (Graphs Banao)
# =============================================================
# Graphs se hum DEKHTE hain ke kaunsa feature price ko
# zyada affect karta hai. Agar dots ek line mein hain,
# toh strong relationship hai.

print("\n" + "=" * 60)
print("📌 STEP 2: Data Visualization")
print("=" * 60)

# --- Graph 1: Area vs Price ---
# Socho: Bada ghar = zyada price? Dekhte hain!
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st graph
plt.scatter(data['Area'], data['Price'], color='blue', alpha=0.7)
plt.xlabel('Area (sq ft)')    # X-axis label
plt.ylabel('Price ($)')       # Y-axis label
plt.title('Area vs Price')    # Graph ka title

# --- Graph 2: Bedrooms vs Price ---
plt.subplot(1, 3, 2)  # 2nd graph
plt.scatter(data['Bedrooms'], data['Price'], color='green', alpha=0.7)
plt.xlabel('Bedrooms')
plt.ylabel('Price ($)')
plt.title('Bedrooms vs Price')

# --- Graph 3: Age vs Price ---
# Purana ghar = sasta? Dekhte hain!
plt.subplot(1, 3, 3)  # 3rd graph
plt.scatter(data['Age'], data['Price'], color='red', alpha=0.7)
plt.xlabel('Age (years)')
plt.ylabel('Price ($)')
plt.title('Age vs Price')

plt.tight_layout()  # Graphs ko properly arrange karo
plt.savefig("step2_scatter_plots.png", dpi=100)  # Graph save karo
print("✅ Scatter plots saved: step2_scatter_plots.png")
plt.show()

# =============================================================
# 📌 STEP 3: Linear Regression FROM SCRATCH (Gradient Descent)
# =============================================================
# Ab hum KHUD se model banayenge!
#
# 🧠 CONCEPT:
# Linear Regression kehta hai:
#   predicted_price = m * area + b
#
# "m" = slope (kitna zyada area se price badhta hai)
# "b" = intercept (base price jab area = 0)
#
# 🎯 GOAL:
# Aise m aur b dhundho ke prediction actual price ke
# CLOSEST ho. Isko hum MSE (Mean Squared Error) se measure
# karte hain.
#
# ⚙️ GRADIENT DESCENT:
# Yeh ek algorithm hai jo m aur b ko DHEERE DHEERE theek
# karta hai. Har baar thoda sa adjust karo, jab tak error
# kam na ho jaye.

print("\n" + "=" * 60)
print("📌 STEP 3: Linear Regression from Scratch")
print("=" * 60)

# --- Step 3a: Data Prepare Karo ---
# Pehle sirf Area use karenge (simple samajhne ke liye)
X = data['Area'].values   # Input feature (independent variable)
y = data['Price'].values   # Output target (dependent variable)

# --- Feature Scaling (BOHAT ZAROORI!) ---
# Area ki values (1000-3000) aur Price ki values (180000-580000)
# bahut badi hain. Gradient Descent ko CHOTI numbers chahiye
# warna yeh "explode" ho jayega (numbers infinity mein chale jayenge).
#
# Normalization formula: x_normalized = (x - mean) / std
# Isse sab values -2 se +2 ke beech aa jaati hain.

X_mean = np.mean(X)     # Area ka average
X_std = np.std(X)        # Area ka standard deviation
y_mean = np.mean(y)      # Price ka average
y_std = np.std(y)        # Price ka standard deviation

X_norm = (X - X_mean) / X_std   # Normalized area
y_norm = (y - y_mean) / y_std   # Normalized price

print(f"Area: mean={X_mean:.0f}, std={X_std:.0f}")
print(f"Price: mean={y_mean:.0f}, std={y_std:.0f}")

# --- Step 3b: Train/Test Split (MANUALLY) ---
# Data ko 2 hisson mein baanto:
# - Training set (80%): Isse model SEEKHEGA
# - Testing set (20%): Isse hum CHECK karenge ke seekha bhi ya nahi

n = len(X_norm)
split = int(0.8 * n)  # 80% = training

# Shuffle karo taa ke data random order mein ho
np.random.seed(42)  # Seed = same random numbers har baar
indices = np.random.permutation(n)

train_idx = indices[:split]  # Pehle 80% indices
test_idx = indices[split:]   # Baaki 20% indices

X_train = X_norm[train_idx]
y_train = y_norm[train_idx]
X_test = X_norm[test_idx]
y_test = y_norm[test_idx]

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- Step 3c: Gradient Descent Algorithm ---
# Yeh hai CORE ML algorithm! Har line samjho:

m = 0.0   # Slope — shuru mein 0 se start
b = 0.0   # Intercept — shuru mein 0 se start

learning_rate = 0.01   # Alpha (α) — kitna bada step lena hai
                        # Bohat bada = overshoot, bohat chota = slow
iterations = 1000      # Kitni baar parameters update karne hain

n_train = len(X_train)
cost_history = []  # Har iteration ka error store karenge

print(f"\n⚙️ Gradient Descent shuru ho raha hai...")
print(f"   Learning Rate (α): {learning_rate}")
print(f"   Iterations: {iterations}")

for i in range(iterations):
    # Step 1: PREDICT karo current m aur b se
    y_predicted = m * X_train + b
    
    # Step 2: ERROR nikalo (kitna door hai prediction actual se)
    error = y_train - y_predicted
    
    # Step 3: COST calculate karo (MSE = Mean Squared Error)
    # MSE = (1/n) * sum((actual - predicted)^2)
    cost = (1 / n_train) * np.sum(error ** 2)
    cost_history.append(cost)
    
    # Step 4: GRADIENTS nikalo
    # Gradient batata hai ke m aur b ko KIDHAR adjust karna hai
    dm = (-2 / n_train) * np.sum(X_train * error)  # m ke liye gradient
    db = (-2 / n_train) * np.sum(error)             # b ke liye gradient
    
    # Step 5: PARAMETERS UPDATE karo
    # Naya m = purana m - learning_rate * gradient
    m = m - learning_rate * dm
    b = b - learning_rate * db
    
    # Har 100 iterations pe progress dikhao
    if (i + 1) % 100 == 0:
        print(f"   Iteration {i+1:4d} | Cost (MSE): {cost:.6f} | m: {m:.4f} | b: {b:.4f}")

print(f"\n✅ Training Complete!")
print(f"   Final m (slope): {m:.4f}")
print(f"   Final b (intercept): {b:.4f}")

# --- Cost History Plot ---
# Yeh graph dikhata hai ke error KAISE kam hota gaya
plt.figure(figsize=(8, 4))
plt.plot(cost_history, color='purple')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Training Progress: Cost Decreasing Over Time')
plt.grid(True, alpha=0.3)
plt.savefig("step3_cost_history.png", dpi=100)
print("✅ Cost history plot saved: step3_cost_history.png")
plt.show()

# =============================================================
# 📌 STEP 4: Model Evaluate Karo (Test Results)
# =============================================================
# Ab dekhte hain ke model ne SEEKHA kya:
# Test data pe predict karo aur actual se compare karo

print("\n" + "=" * 60)
print("📌 STEP 4: Model Evaluation")
print("=" * 60)

# Test data pe prediction (normalized values pe)
y_test_pred_norm = m * X_test + b

# Predictions ko WAPAS original scale pe lao (de-normalize)
y_test_pred = y_test_pred_norm * y_std + y_mean
y_test_actual = y_test[:]  * y_std + y_mean  # Actual bhi de-normalize
X_test_actual = X_test * X_std + X_mean      # Area bhi de-normalize

# --- MSE (Mean Squared Error) ---
# Jitna CHOTA, utna ACHA model
mse = np.mean((y_test_actual - y_test_pred) ** 2)

# --- R² Score ---
# 1.0 = PERFECT model, 0 = bekar, negative = bohat bekar
# Formula: 1 - (sum of squared errors / total variance)
ss_res = np.sum((y_test_actual - y_test_pred) ** 2)  # Residual
ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)  # Total
r2_score = 1 - (ss_res / ss_tot)

print(f"\n📊 Test Results:")
print(f"   Mean Squared Error (MSE): {mse:,.0f}")
print(f"   Root MSE (RMSE):          ${np.sqrt(mse):,.0f}")
print(f"   R² Score:                  {r2_score:.4f}")
print(f"   (R² = 1.0 matlab perfect, 0.9+ matlab bohat acha)")

# --- Actual vs Predicted Table ---
print(f"\n📋 Actual vs Predicted Prices:")
print(f"   {'Area':>8} | {'Actual':>10} | {'Predicted':>10} | {'Difference':>10}")
print(f"   {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")
for i in range(len(y_test_actual)):
    diff = y_test_actual[i] - y_test_pred[i]
    print(f"   {X_test_actual[i]:>8.0f} | ${y_test_actual[i]:>9,.0f} | ${y_test_pred[i]:>9,.0f} | ${diff:>+9,.0f}")

# --- Predicted vs Actual Plot ---
plt.figure(figsize=(10, 5))

# Left plot: regression line on test data
plt.subplot(1, 2, 1)
plt.scatter(X_test_actual, y_test_actual, color='blue', label='Actual', s=80)
plt.scatter(X_test_actual, y_test_pred, color='red', marker='x', label='Predicted', s=80)

# Regression line draw karo
x_line = np.linspace(X.min(), X.max(), 100)
x_line_norm = (x_line - X_mean) / X_std
y_line_norm = m * x_line_norm + b
y_line = y_line_norm * y_std + y_mean
plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.title('Test Data: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

# Right plot: predicted vs actual (perfect = diagonal line)
plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_test_pred, color='green', s=80)
# Perfect prediction line
min_val = min(y_test_actual.min(), y_test_pred.min())
max_val = max(y_test_actual.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Predicted vs Actual (closer to red line = better)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("step4_results.png", dpi=100)
print("\n✅ Results plot saved: step4_results.png")
plt.show()

# =============================================================
# 🎉 SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("🎉 CONGRATULATIONS! Tumne apna pehla ML model bana liya!")
print("=" * 60)
print(f"""
Tumne yeh seekha:
  ✅ Data kaise load aur explore karte hain
  ✅ Scatter plots se relationships kaise dekhte hain
  ✅ Linear Regression ANDAR se kaise kaam karta hai
  ✅ Gradient Descent kya hai aur kaise m, b update hote hain
  ✅ MSE aur R² Score se model evaluate karte hain

Ab 'linear_regression_sklearn.py' chalao aur dekho ke
scikit-learn yeh sab AUTOMATICALLY kaise karta hai! 🚀
""")
