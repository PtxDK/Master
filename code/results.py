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
    return pd.concat(dfs).drop(columns=["epoch"])


#%%
#%%

mi: pd.DataFrame = load_glob("./logs/UNet3D_aug-*-[012]-evaluate.csv"
                            ).set_index(["dim", "prob", "alpha", "i"])
#%%
print(mi.groupby(["dim", "prob", "alpha"]).agg('std').idxmin())
print(mi.groupby(["dim", "prob", "alpha"]).agg(['mean']).idxmin())

#%%
for i in ["dim", "prob", "alpha"]:
    print(mi.groupby([i]).agg(['mean']))
# %%
idxs = [["hyper", "augmented", "pad", "shrink", "base"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
data = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.split("./logs\\")[-1].split("-final-evaluate.csv")[0]]
        ) for i in files
    ]
).drop(columns=["epoch"])
data.index = idxs
print(data.groupby(["model"]).agg(['mean']))
#%%
idxs = [
    ["hyper", "augmentation", "pad", "hyper2", "shrink", "base"],
    list(range(3))
]
idxs = pd.MultiIndex.from_product(idxs, names=["model", "i"])
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
# files = [i.replace("-final", "") for i in files]
print(files)
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
mpu = pd.read_csv("./logs/mpu.csv").set_index(["model"])
idxs = [("MPUNet", 0), ("MPUNet", 1), ("MPUNet", 2),]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
mpu.index = idxs
print(mpu.groupby(["model"]).agg(["mean", "std"]).to_latex(float_format="%.3f"))
#%%
mpu = pd.read_csv("./logs/mpu-noaug.csv").set_index(["model"])
idxs = [("MPUNet", 0), ("MPUNet", 1), ("MPUNet", 2),]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
mpu.index = idxs
print(mpu.groupby(["model"]).agg(["mean", "std"]).to_latex(float_format="%.3f"))
#%%
files = glob.glob("./logs/UNet3D_augmentation-*-final-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.split("./logs\\")[-1].split("-final-evaluate.csv")[0]]
        ) for i in files
    ]
).drop(columns=["epoch"])
idxs = [["aug"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["model", "i"])
data1.index = idxs
print(data1.groupby(["model"]).agg(['mean', 'std']))
#%%
files = glob.glob("./logs/UNet3D_hyper-*[012]-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.split("./logs\\")[-1].split("-final-evaluate.csv")[0]]
        ) for i in files
    ]
).drop(columns=["epoch"]).sort_index()
idxs = [
    (1, 3, 0), (1, 3, 1), (1, 3, 2), (1, 4, 0), (1, 4, 1), (1, 4, 2), (1, 5, 0),
    (1, 5, 1), (1, 5, 2), (2, 3, 0), (2, 3, 1), (2, 3, 2)
]
idxs = pd.MultiIndex.from_tuples(idxs, names=["cf", "depth", "i"])
data1.index = idxs
print(data1.groupby(["cf", "depth"]).agg(['mean']).idxmax())
#%%
files = glob.glob("./logs/UNet3D_hyper-20-30-1-4-[012]-final-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.replace("./logs\\", "").replace("-final-evaluate.csv", "")]
        ) for i in files
    ]
).drop(columns=["epoch"]).sort_index()
print(data1.agg(['mean']))
#%%
files = glob.glob("./logs/UNet3D_augmentation-[012]-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.replace("./logs/", "").replace("-evaluate.csv", "")]
        ) for i in files
    ]
).drop(columns=["epoch"]).sort_index()
idxs = [
    ("UNet3D_augmentation", 0), ("UNet3D_augmentation", 1),
    ("UNet3D_augmentation", 2),
]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
data1.index = idxs
print(
    data1.groupby(["model"]).agg(['mean', 'std']).to_latex(float_format="%.3f")
)
#%%
files = glob.glob("./logs/UNet_base[012]-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.replace("./logs/", "").replace("-evaluate.csv", "")]
        ) for i in files
    ]
).drop(columns=["epoch"]).sort_index()
idxs = [("2D U-Net", 0), ("2D U-Net", 1), ("2D U-Net", 2),]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
data1.index = idxs
print(
    data1.groupby(["model"]).agg(['mean', 'std']).to_latex(float_format="%.3f")
)
#%%
files = glob.glob("./logs/UNet_base[012]-final-evaluate.csv")
data1 = pd.concat(
    [
        pd.read_csv(i).set_axis(
            [i.replace("./logs/", "").replace("-evaluate.csv", "")]
        ) for i in files
    ]
).drop(columns=["epoch"]).sort_index()
idxs = [("2D U-Net", 0), ("2D U-Net", 1), ("2D U-Net", 2),]
idxs = pd.MultiIndex.from_tuples(idxs, names=["model", "i"])
data1.index = idxs
print(
    data1.groupby(["model"]).agg(['mean', 'std']).to_latex(float_format="%.3f")
)
