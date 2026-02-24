"""
STEP 5: Train XGBoost + LSTM to predict excess job exits after disasters

What this does:
1. Loads the sector-level feature matrix (3,417 rows, 63 features)
2. Splits by disaster event (GroupKFold) — model never sees the same
   disaster in both train and test, preventing data leakage
3. Trains XGBoost (tabular model — 500 boosted decision trees)
4. Trains LSTM (sequential model — reads monthly exits as a time series)
5. Compares both on MAE, RMSE, R² and picks the winner
6. Saves the best model + all predictions for the API
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

df = pd.read_csv("features.csv")

with open("feature_columns.txt") as f:
    feature_cols = [line.strip() for line in f.readlines() if line.strip()]

monthly_cols = [f'exits_month_{i}' for i in range(1, 7)]
target_col = 'excess_exits'

X = df[feature_cols].copy()
y = df[target_col].copy()
groups = df['disasterNumber'].values

print(f"Loaded {len(df)} rows, {len(feature_cols)} features")
print(f"Target (excess_exits): mean={y.mean():.2f}, std={y.std():.2f}")
print(f"Unique disasters (groups): {len(np.unique(groups))}")
print(f"Monthly columns for LSTM: {monthly_cols}")


# XGBOOST
print(f"\n{'='*60}")
print(f"XGBOOST — 5-FOLD GROUP CROSS-VALIDATION")
print(f"{'='*60}")

n_folds = 5
gkf = GroupKFold(n_splits=n_folds)

xgb_fold_results = []
xgb_all_preds = np.zeros(len(df))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    xgb_all_preds[test_idx] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    n_train = len(np.unique(groups[train_idx]))
    n_test = len(np.unique(groups[test_idx]))

    xgb_fold_results.append({'fold': fold+1, 'mae': mae, 'rmse': rmse, 'r2': r2})
    print(f"  Fold {fold+1}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  "
          f"(train: {n_train} disasters, test: {n_test} disasters)")

xgb_results = pd.DataFrame(xgb_fold_results)
print(f"\nXGBoost Average:")
print(f"  MAE:  {xgb_results['mae'].mean():.3f} ± {xgb_results['mae'].std():.3f}")
print(f"  RMSE: {xgb_results['rmse'].mean():.3f} ± {xgb_results['rmse'].std():.3f}")
print(f"  R²:   {xgb_results['r2'].mean():.3f} ± {xgb_results['r2'].std():.3f}")

# Train final XGBoost on all data
final_xgb = xgb.XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
)
final_xgb.fit(X, y, verbose=False)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Features:")
for _, row in importance.head(15).iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:40s} {row['importance']:.4f}  {bar}")


# LSTM
print(f"\n{'='*60}")
print(f"LSTM — 5-FOLD GROUP CROSS-VALIDATION")
print(f"{'='*60}")

# Separate static features from sequence features
# Static = all features EXCEPT the monthly exit columns
# Sequence = exits_month_1 through exits_month_6 (6 time steps)
static_cols = [c for c in feature_cols if c not in monthly_cols]

# Check if monthly cols are in feature_cols (they might be excluded)
seq_cols_in_features = [c for c in monthly_cols if c in feature_cols]
if not seq_cols_in_features:
    # Monthly cols aren't in feature_cols — read them directly from df
    seq_cols_in_features = monthly_cols

print(f"Static features: {len(static_cols)}")
print(f"Sequence steps: {len(seq_cols_in_features)} (monthly exits)")


class DisasterLSTM(nn.Module):
    """LSTM that reads monthly exit sequence + static features to predict excess exits."""

    def __init__(self, n_static, n_seq_features=1, hidden_size=64, n_layers=2):
        super().__init__()
        # LSTM reads the monthly exit sequence (6 steps, 1 feature per step)
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2,
        )
        # Combine LSTM output with static features → prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + n_static, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq, static):
        # seq shape: (batch, 6, 1) — 6 months, 1 value per month
        # static shape: (batch, n_static)
        lstm_out, (h_n, _) = self.lstm(seq)
        # Take the last hidden state (final memory after reading all 6 months)
        last_hidden = h_n[-1]  # shape: (batch, hidden_size)
        # Concatenate with static features
        combined = torch.cat([last_hidden, static], dim=1)
        return self.fc(combined).squeeze(-1)


def train_lstm_fold(X_train_static, X_train_seq, y_train,
                    X_test_static, X_test_seq, y_test,
                    n_epochs=100, lr=0.001, batch_size=128):
    """Train LSTM for one fold and return predictions."""

    # Scale features
    scaler_static = StandardScaler()
    X_train_s = scaler_static.fit_transform(X_train_static)
    X_test_s = scaler_static.transform(X_test_static)

    scaler_seq = StandardScaler()
    X_train_q = scaler_seq.fit_transform(X_train_seq)
    X_test_q = scaler_seq.transform(X_test_seq)

    # Convert to tensors
    train_static_t = torch.FloatTensor(X_train_s)
    train_seq_t = torch.FloatTensor(X_train_q).unsqueeze(-1)  # (N, 6, 1)
    train_y_t = torch.FloatTensor(y_train.values)

    test_static_t = torch.FloatTensor(X_test_s)
    test_seq_t = torch.FloatTensor(X_test_q).unsqueeze(-1)
    test_y_t = torch.FloatTensor(y_test.values)

    # DataLoader
    train_dataset = TensorDataset(train_seq_t, train_static_t, train_y_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = DisasterLSTM(n_static=X_train_s.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for seq_batch, static_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(seq_batch, static_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_preds = model(test_seq_t, test_static_t).numpy()

    return test_preds


# Run LSTM with same GroupKFold splits
lstm_fold_results = []
lstm_all_preds = np.zeros(len(df))

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train_static = df[static_cols].iloc[train_idx]
    X_test_static = df[static_cols].iloc[test_idx]

    X_train_seq = df[seq_cols_in_features].iloc[train_idx]
    X_test_seq = df[seq_cols_in_features].iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    preds = train_lstm_fold(
        X_train_static, X_train_seq, y_train,
        X_test_static, X_test_seq, y_test,
        n_epochs=100, lr=0.001, batch_size=128,
    )

    lstm_all_preds[test_idx] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    n_train = len(np.unique(groups[train_idx]))
    n_test = len(np.unique(groups[test_idx]))

    lstm_fold_results.append({'fold': fold+1, 'mae': mae, 'rmse': rmse, 'r2': r2})
    print(f"  Fold {fold+1}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  "
          f"(train: {n_train} disasters, test: {n_test} disasters)")

lstm_results = pd.DataFrame(lstm_fold_results)
print(f"\nLSTM Average:")
print(f"  MAE:  {lstm_results['mae'].mean():.3f} ± {lstm_results['mae'].std():.3f}")
print(f"  RMSE: {lstm_results['rmse'].mean():.3f} ± {lstm_results['rmse'].std():.3f}")
print(f"  R²:   {lstm_results['r2'].mean():.3f} ± {lstm_results['r2'].std():.3f}")


# COMPARISON
print(f"\n{'='*60}")
print(f"MODEL COMPARISON")
print(f"{'='*60}")

# Baseline: always predict the mean
baseline_mae = mean_absolute_error(y, np.full(len(y), y.mean()))

xgb_mae = xgb_results['mae'].mean()
lstm_mae = lstm_results['mae'].mean()

print(f"  Baseline (predict mean):  MAE={baseline_mae:.3f}")
print(f"  XGBoost:                  MAE={xgb_mae:.3f}  RMSE={xgb_results['rmse'].mean():.3f}  R²={xgb_results['r2'].mean():.3f}")
print(f"  LSTM:                     MAE={lstm_mae:.3f}  RMSE={lstm_results['rmse'].mean():.3f}  R²={lstm_results['r2'].mean():.3f}")
print(f"")
print(f"  XGBoost vs baseline: {((baseline_mae - xgb_mae) / baseline_mae * 100):.1f}% improvement")
print(f"  LSTM vs baseline:    {((baseline_mae - lstm_mae) / baseline_mae * 100):.1f}% improvement")

winner = "XGBoost" if xgb_mae < lstm_mae else "LSTM"
print(f"\n  Winner: {winner}")


# ══════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════

# Save XGBoost model (always save — it's the reliable one)
joblib.dump(final_xgb, "xgb_model.joblib")
print(f"\nSaved XGBoost model to data/xgb_model.joblib")

# Save predictions from both models
df['xgb_predicted'] = xgb_all_preds
df['lstm_predicted'] = lstm_all_preds
df.to_csv("predictions.csv", index=False)
print(f"Saved predictions to data/predictions.csv")

# Save feature importance
importance.to_csv("feature_importance.csv", index=False)
print(f"Saved feature importance to data/feature_importance.csv")

print(f"\n{'='*60}")
print(f"DONE — Ready for Step 6 (predictions export)")
print(f"{'='*60}")
