#%%
import pandas as pd
import glob


def load_glob(val):
    files = glob.glob(val)
    dfs = []
    for i in files:
        splits = i.split("-")
        vals = {
            "dim": splits[1],
            "prob": splits[2],
            "alpha": splits[3],
            "i": splits[4]
        }

        dfs.append(pd.read_csv(i).join(pd.DataFrame(pd.Series(vals)).T))
    return pd.concat(dfs).drop(columns=["epoch", "fg_f1"])


mi: pd.DataFrame = load_glob("./logs/UNet3D_aug-*-[012]-evaluate.csv"
                            ).set_index(["dim", "prob", "alpha", "i"])
#%%
print(mi.sort_index().to_latex(float_format="%.3f"))
# print(mi.groupby(["dim", "prob", "alpha"]).agg('std').idxmin())
# print(mi.groupby(["dim", "prob", "alpha"]).agg(['mean']).idxmin())

#%%
for i in ["dim", "prob", "alpha"]:
    print(mi.groupby([i]).agg(['mean']))
# %%
idxs = [
    [
        "3D U-Net H.P. tuning set 1", "3D U-Net, D.A.",
        "3D U-Net H.P. tuning set 2", "3D U-Net, padding",
        "3D U-Net, shrinking", "2D U-Net", "MPUNet", "MPUNet, no D.A."
    ],
    list(range(3))
]
idxs = pd.MultiIndex.from_product(idxs, names=["model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
files = [i for i in files if "pad-normal" not in i]
data = pd.concat(
    [
        pd.read_csv(i) for i in files
    ]
).drop(columns=["epoch", "fg_f1", "model"])

data.index = idxs
print(data.groupby(["model"]).agg('mean').to_latex(float_format="%.3f"))
print(data.to_latex(float_format="%.3f"))
#%%
files = glob.glob("./logs/UNet3D_hyper-*[012]-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.split("./logs\\")[-1].split("-final-evaluate.csv")[0]]
        ) for i in files
    ]
).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [
    (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 5, 0),
    (1, 5, 1), (1, 5, 2), (2, 3, 0), (2, 3, 1), (2, 3, 2)
]
idxs = pd.MultiIndex.from_tuples(idxs, names=["cf", "depth", "i"])
data1.index = idxs
print(data1.to_latex(float_format="%.3f"))
#%%
files = glob.glob("./logs/UNet3D_augmentation-[012]-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.replace("./logs/", "").replace("-evaluate.csv", "")]
        ) for i in files
    ]
).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [
    ("UNet3D_augmentation", 0), ("UNet3D_augmentation", 1),
    ("UNet3D_augmentation", 2),
]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
data1.index = idxs
print(data1.groupby(["model"]).agg('mean'))
#%%
files = glob.glob("./logs/UNet_base[012]-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.replace("./logs/", "").replace("-evaluate.csv", "")]
        ) for i in files
    ]
).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [("2D U-Net", 0), ("2D U-Net", 1), ("2D U-Net", 2),]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
data1.index = idxs
print(data1.groupby(["model"]).agg('mean'))
#%%
genders = [
    "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "F", "F",
    "F", "M", "F", "M", "M", "F", "F", "F", "M", "M", "M", "M", "M", "F", "F",
    "F", "M", "M", "M", "M", "M", "M", "M", "F", "F", "M", "M", "F", "M", "M",
    "F", "M", "F", "M", "M", "M", "M", "M", "F", "M", "F", "M", "F", "F", "M",
    "M"
]
splits = [("train", 26), ("val", 9), ("test", 9), ("true_test", 17)]
start = 0
curr = 0
dict_val = {}
for (name, idx) in splits:
    curr = idx
    men = len([i for i in genders[start:curr] if i == "M"])
    women = len([i for i in genders[start:curr] if i == "F"])
    dict_val[name] = [men, women]
dict_val["total"] = [
    len([i for i in genders if i == "M"]),
    len([i for i in genders if i == "F"])
]
data = pd.DataFrame(dict_val).set_axis(["M", "F"]).T
res = data.div(data.sum(axis=1), axis=0)
print(res)
# %%
