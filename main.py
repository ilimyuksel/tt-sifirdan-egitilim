import torch
import polars as pl
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pl.read_csv("./house_price_regression_dataset.csv")

######## normalize
cols_to_normalize = (
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms", 
    "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality",
    "House_Price"
)


saved_means = {

}

saved_devs = {

}

cols_exprs = []
for c in cols_to_normalize:
    saved_means[c] = df.select(pl.col(c).mean()).item()
    saved_devs[c] = df.select(pl.col(c).std()).item()

    cols_exprs.append(
        ((pl.col(c)-pl.col(c).mean()) / pl.col(c).std())
    )


df = df.with_columns(cols_exprs)

########## Model

class MyHouseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

#### Train/test split

X = df.drop("House_Price")

y = df[["House_Price"]]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)


###### Dataset/DataLoader

class MyHouseDataset(Dataset):
    def __init__(self, X_df: pl.DataFrame, y_df):
        self.X_arr = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
        self.y_arr = torch.tensor(y_df.to_numpy(), dtype=torch.float32)

    def __getitem__(self, idx: int):
        return self.X_arr[idx], self.y_arr[idx]

    def __len__(self) -> int:
        return self.X_arr.shape[0]


train_ds = MyHouseDataset(X_train, y_train)
val_ds = MyHouseDataset(X_val, y_val)


train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)

##### Loss_fn & optimizer


model = MyHouseModel(input_dim=7, output_dim=1)
loss_fn = nn.HuberLoss()
optim = Adam(model.parameters(), lr=0.001)

### Train/Eval loops

EPOCHS = 30
for e in range(EPOCHS):
    train_loss = 0
    for X,y in train_dl:
        optim.zero_grad()

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        train_loss += float(loss)

        loss.backward()
        optim.step()

    
    val_loss = 0
    for X,y in val_dl:
        with torch.no_grad():
     
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            val_loss += float(loss)


    train_loss /= len(train_dl)
    val_loss /= len(val_dl)

    print(f"EPOCH {e} || TRAIN LOSS {train_loss:.5f} || VAL LOSS {val_loss:.5f}")
    



#### EVAL
preds = []
gt = []

for X,y in val_ds:
    with torch.no_grad():
        y_pred = float(model(X))

    y_pred = (y_pred*saved_devs["House_Price"]) + saved_means["House_Price"]
    y_gt = (y*saved_devs["House_Price"]) + saved_means["House_Price"]

    preds.append(y_pred)
    gt.append(y_gt)


metric = mean_absolute_error(gt, preds)
print(f"MAE {metric:.2f}")
    




