#%%
import pandas as pd
import glob


def load_glob(val):
    files = glob.glob(val)
    dfs = []
    for i in files:
        splits = i.split("-")
        vals = {
            "Dimension": splits[1],
            "Probability": splits[2],
            "Alpha": splits[3],
            "i": splits[4]
        }

        dfs.append(pd.read_csv(i).join(pd.DataFrame(pd.Series(vals)).T))
    return pd.concat(dfs).drop(columns=["epoch", "fg_f1"])


mi: pd.DataFrame = load_glob("./logs/UNet3D_aug-*-[012]-evaluate.csv"
                            ).set_index(
                                ["Dimension", "Probability", "Alpha", "i"]
                            )
mi = mi.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)
#%%
print(
    mi.groupby(['Dimension', 'Probability',
                'Alpha']).agg(['mean', 'std']).to_latex(float_format="%.3f")
)
#%%
print(mi.sort_index().to_latex(float_format="%.3f"))
#%%
for i in ["Dimension", "Probability", "Alpha"]:
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
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
files = [i for i in files if "pad-normal" not in i and "scaled" not in i]
eval_res = pd.concat([pd.read_csv(i) for i in files]
                    ).drop(columns=["epoch", "fg_f1", "model"])

eval_res.index = idxs
eval_res = eval_res.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)
print(
    eval_res.groupby(["Model"]).agg(['mean',
                                     'std']).to_latex(float_format="%.3f")
)
#%%
print(eval_res.to_latex(float_format="%.3f"))

#%%
# files = glob.glob("./logs/*-final-evaluate.csv")
# files.sort()
files = [
    i.replace("-final", "")
    for i in files
    if "pad-normal" not in i and "noaug" not in i
]
data1 = pd.concat([pd.read_csv(i) for i in files]
                 ).drop(columns=["epoch", "fg_f1", "model"]).sort_index()

idxs = [
    [
        "3D U-Net H.P. tuning set 1", "3D U-Net, D.A.",
        "3D U-Net H.P. tuning set 2", "3D U-Net, padding",
        "3D U-Net, shrinking", "2D U-Net", "MPUNet"
    ],
    list(range(3))
]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
data1.index = idxs
data1 = data1.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)
print(
    data1[["Dice", "Precision",
           "Recall"]].groupby("Model").agg(['mean', 'std']
                                          ).to_latex(float_format="%.3f")
)
#%%
print(data1.to_latex(float_format="%.3f"))
#%%
files = glob.glob("./logs/UNet3D_hyper-*[012]-evaluate.csv")
test_res = pd.concat(
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
idxs = pd.MultiIndex.from_tuples(
    idxs, names=["Complexity Factor", "Depth", "i"]
)
test_res.index = idxs
test_res = test_res.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)
print(test_res.to_latex(float_format="%.3f"))
#%%
from collections import Counter
genders = [
    "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "F", "F",
    "F", "M", "F", "M", "M", "F", "F", "F", "M", "M", "M", "M", "M", "F", "F",
    "F", "M", "M", "M", "M", "M", "M", "M", "F", "F", "M", "M", "F", "M", "M",
    "F", "M", "F", "M", "M", "M", "M", "M", "F", "M", "F", "M", "F", "F", "M",
    "M"
]
splits = [("Train", 26), ("Validation", 9), ("Test", 9), ("Evaluation", 17)]
curr = 0
dict_val = {}
for (name, add_val) in splits:
    counts = Counter(genders[curr:curr + add_val])
    curr += add_val
    dict_val[name] = counts
counts = Counter(genders)
dict_val["total"] = counts
data = pd.DataFrame(dict_val).set_axis(["M", 'F']).T
data[["M%", "F%"]] = (data.iloc[:, :] / data.sum(axis=1)[:, None])
data = data.sort_index(axis=1)
print(data.to_latex(float_format="%.3f"))
# %%
