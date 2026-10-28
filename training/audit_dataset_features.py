import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

data = np.load('data/processed/realsign_landmarks_26class.npz')
X = data['X']
y = data['y']
splits = data['splits']
classes = data['classes']

test_mask = (splits == 'test')
X_test = X[test_mask]
y_test = y[test_mask]

print(f"Total samples: {len(y)}")
print(f"Total test samples: {len(y_test)}")

# 1. Zero-variance features
vars_ = np.var(X, axis=0)
zero_var_idx = np.where(vars_ < 1e-6)[0]
print(f"\n[1] Zero variance feature indices ({len(zero_var_idx)} total):")
for idx in zero_var_idx:
    lm_idx = idx // 3
    coord = ['x', 'y', 'z'][idx % 3]
    print(f"  Feature {idx:02d}: Landmark {lm_idx:02d} {coord} = {X[0, idx]:.4f} (Var = {vars_[idx]:.8f})")

# 2. Extreme collinearity
corr = np.corrcoef(X.T)
np.fill_diagonal(corr, 0)
corr[np.isnan(corr)] = 0
high_corr = np.where(np.abs(corr) > 0.95)
high_corr_pairs = [(i, j, float(corr[i, j])) for i, j in zip(high_corr[0], high_corr[1]) if i < j]
print(f"\n[2] High Collinearity Pairs (|r| > 0.95): {len(high_corr_pairs)} pairs")
for i, j, r in high_corr_pairs[:10]:
    lm_i, coord_i = i // 3, ['x', 'y', 'z'][i % 3]
    lm_j, coord_j = j // 3, ['x', 'y', 'z'][j % 3]
    print(f"  Feature {i:02d} (LM {lm_i} {coord_i}) <-> Feature {j:02d} (LM {lm_j} {coord_j}): r = {r:.4f}")

# 3. Model evaluation & Confusions
class RealSignMLP(nn.Module):
    def __init__(self, in_features=63, hidden1=128, hidden2=64, num_classes=26, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )
    def forward(self, x):
        return self.net(x)

model = RealSignMLP()
checkpoint = torch.load('models/weights/realsign_26class_mlp.pth', map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()

cm = confusion_matrix(y_test, preds)
print("\n[3] Top 15 Most Confused Pairs in Test Set:")
confusions = []
for i in range(26):
    for j in range(26):
        if i != j and cm[i, j] > 0:
            confusions.append((classes[i], classes[j], int(cm[i, j])))
confusions.sort(key=lambda x: x[2], reverse=True)
for true_c, pred_c, count in confusions[:15]:
    print(f"  True: {true_c} -> Predicted: {pred_c} ({count} errors)")

report = classification_report(y_test, preds, target_names=classes, output_dict=True)
lowest_f1 = sorted([(c, report[c]['f1-score'], report[c]['recall'], report[c]['precision'], report[c]['support']) for c in classes], key=lambda x: x[1])
print("\n[4] Lowest 8 F1 Classes:")
for c, f1, rec, prec, sup in lowest_f1[:8]:
    print(f"  Class {c}: F1 = {f1*100:.2f}%, Recall = {rec*100:.2f}%, Precision = {prec*100:.2f}%, Support = {sup}")

# 5. Check depth / z coordinate variance vs x/y variance
x_vars = [vars_[i] for i in range(0, 63, 3) if i not in zero_var_idx]
y_vars = [vars_[i] for i in range(1, 63, 3) if i not in zero_var_idx]
z_vars = [vars_[i] for i in range(2, 63, 3) if i not in zero_var_idx]
print(f"\n[5] Average Coordinate Variance Across Landmarks:")
print(f"  Mean X-variance: {np.mean(x_vars):.5f}")
print(f"  Mean Y-variance: {np.mean(y_vars):.5f}")
print(f"  Mean Z-variance: {np.mean(z_vars):.5f} (monocular depth camera noise!)")
