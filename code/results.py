#%%
import pandas as pd
import glob

#%%
files = glob.glob("./logs/UNet3D_aug-*-evaluate.csv")
dfs = []
for i in files:
    # {dim}-{prob}-{alpha}-{i}
    splits = i.split("-")
    vals = {
        "dim": splits[1],
        "prob": splits[2],
        "alpha": splits[3],
        "i": splits[4]
    }

    dfs.append(pd.read_csv(i).join(pd.DataFrame(pd.Series(vals)).T))
#%%

data = pd.concat(dfs).drop(columns=["epoch"])
mi: pd.DataFrame = data.set_index(["dim", "prob", "alpha", "i"])
#%%
mi.groupby(["dim", "prob", "alpha"]).agg('std').idxmin()
mi.groupby(["dim", "prob", "alpha"]).agg('mean').idxmax()

#%%
for i in ["dim", "prob", "alpha"]:
    print(mi.groupby([i]).agg(['mean']))
#%%