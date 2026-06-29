import torch
import polars as pl
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert.eval()

######## data

df = pl.read_csv("./IMDB_TT.csv")
df = df.with_columns(
    (pl.col("sentiment") == "positive").cast(pl.Int64).alias("label")
)

texts = df["review"].to_list()
labels = df["label"].to_list()

######## extract frozen BERT embeddings (CLS token)

BATCH_SIZE = 16
embeddings = []

with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        ).to(DEVICE)

        out = bert(**encoded)
        cls_embeddings = out.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())

X = torch.cat(embeddings, dim=0)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

#### Train/test split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

###### Dataset/DataLoader

class MyReviewDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return self.X.shape[0]


train_ds = MyReviewDataset(X_train, y_train)
val_ds = MyReviewDataset(X_val, y_val)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)

########## Classifier head on top of frozen embeddings

class MySentimentModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = MySentimentModel(input_dim=X.shape[1]).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optim = Adam(model.parameters(), lr=0.001)

### Train/Eval loop with early stopping

EPOCHS = 20
PATIENCE = 3

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0

for e in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_dl:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optim.zero_grad()

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        train_loss += loss.item()

        loss.backward()
        optim.step()

    model.eval()
    val_loss = 0
    all_preds, all_gt = [], []
    for X_batch, y_batch in val_dl:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        with torch.no_grad():
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            val_loss += loss.item()

            all_preds.extend((torch.sigmoid(y_pred) > 0.5).int().cpu().tolist())
            all_gt.extend(y_batch.int().cpu().tolist())

    train_loss /= len(train_dl)
    val_loss /= len(val_dl)
    val_acc = accuracy_score(all_gt, all_preds)

    print(f"EPOCH {e} || TRAIN LOSS {train_loss:.5f} || VAL LOSS {val_loss:.5f} || VAL ACC {val_acc:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {e} (best val loss {best_val_loss:.5f})")
            break

model.load_state_dict(best_state)
