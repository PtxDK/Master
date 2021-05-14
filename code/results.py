#%%
import pandas as pd
import glob


def load_glob(val):
    files = glob.glob(val)
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
    return pd.concat(dfs).drop(columns=["epoch"])


#%%
#%%

mi: pd.DataFrame = load_glob("./logs/UNet3D_aug-*-[012]-evaluate.csv"
                            ).set_index(["dim", "prob", "alpha", "i"])
#%%
print(mi.groupby(["dim", "prob", "alpha"]).agg('std').idxmin())
print(
    mi.groupby(["dim", "prob",
                "alpha"]).agg(['mean', 'std']).to_latex(float_format="%.3f")
)

#%%
for i in ["dim", "prob", "alpha"]:
    print(mi.groupby([i]).agg(['mean']))
# %%
idxs = [["hyper", "pad", "shrink", "base"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
dfs = []
data = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.split("./logs\\")[-1].split("-final-evaluate.csv")[0]]
        ) for i in files
    ]
).drop(columns=["epoch"])
data.index = idxs
print(
    data.groupby(["model"]).agg(['mean', 'std']).to_latex(float_format="%.3f")
)
#%%
idxs = [["hyper", "pad", "shrink", "base"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
files = [i.replace("-final", "") for i in files]
dfs = []
data = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.split("./logs\\")[-1].split("-final-evaluate.csv")[0]]
        ) for i in files
    ]
).drop(columns=["epoch"])
data.index = idxs
print(
    data.groupby(["model"]).agg(['mean', 'std']).to_latex(float_format="%.3f")
)