#%%
import pandas as pd
import glob

latex_kwargs = dict(
    position="H",
    escape=False,
    float_format="%.3f",
    na_rep="-",
    multirow=True,
    multicolumn=True
)


def bold_max(x: pd.Series):
    x = round(x, 3)
    x_ret = x.apply(round_vals)
    if "mean" in x.name and "Loss" not in x.name:
        x_ret[x.idxmax()] = f"\\textbf{ {x[x.idxmax()]} }"
    else:
        x_ret[x.idxmin()] = f"\\textbf{ {x[x.idxmin()]} }"
    return x_ret


def round_vals(x):
    return f"{x:.3f}" if "nan" != f"{x:.3f}" else "-"


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


#%% Hyper Results 1
mi: pd.DataFrame = load_glob("./logs/UNet3D_aug-*-[012]-evaluate.csv"
                            ).set_index(
                                ["Dimension", "Probability", "Alpha", "i"]
                            )
mi = mi.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)

mi.groupby(['Dimension', 'Probability', 'Alpha']).aggregate(
    ['mean', 'std']
).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-1-mean.tex",
    caption=
    "Full results from the first set of hyper parameter tuning over the values of the dimension, alpha upper bound, and application probability",
    label="tab:hyper-1-results"
)
mi.sort_index().applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-1-full.tex",
    caption=
    "Hyper parameter (H.P) search full results from the first set with 3D U-Net using data augmentation",
    label="tab:hyper-1-results-full"
)
#%% Hyper Results 2
files = glob.glob("./logs/UNet3D_hyper-*[012]-evaluate.csv")
test_res = pd.concat([pd.read_csv(i) for i in files]
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
test_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-2-full.tex",
    caption=
    "Full results from the second set of hyper parameter tuning over the values of the complexity factor and depth, the higher complexity factor and depth values ran in to out-of-memory errors",
    label="tab:hyper-2-results-full"
)
test_res = test_res.groupby(["Complexity Factor", "Depth"]).agg(['mean', 'std'])

test_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/hyper-2-mean.tex",
    caption=
    "Hyper parameter (H.P) search results from the second set with 3D U-Net using data augmentation",
    label="tab:hyper-2-results",
)
#%% Test results
files = glob.glob("./logs/*-final-evaluate.csv")
files.sort()
files = [
    i.replace("-final", "")
    for i in files
    if "pad-normal" not in i and "noaug" not in i and "scaled" not in i
]

test_res = pd.concat([pd.read_csv(i) for i in files]
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
test_res.index = idxs
test_res = test_res.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)
test_res_mean = test_res.groupby("Model").agg(['mean', 'std']).apply(bold_max)
test_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-full.tex",
    caption=
    "Full results from the models when tested on the test set",
    label="tab:test-results-full"
)
test_res.loc[[
    "2D U-Net", "3D U-Net, padding", "3D U-Net, shrinking", "MPUNet"
]].groupby("Model").agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-first.tex",
    caption=
    "Early performance results on models created with train and validation data, predicting on testing data. The best performing model appears to be the \"3D U-Net, shrinking\", whereas the worst performing model appears to be \"2D U-Net\"",
    label="tab:test-results-first"
)
test_res_mean.to_latex(
    **latex_kwargs,
    buf="./tables/test-results-mean.tex",
    caption=
    "Performance results from various models, running only with validation data and optimizations.",
    label="tab:test-results"
)
test_res.loc[["3D U-Net, shrinking", "3D U-Net, D.A."]].groupby("Model").agg(
    ['mean', 'std']
).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/test-results-da.tex",
    caption=
    "Early performance results on models with and without Data Augmentation, D.A. stands for data augmentation, while 3D U-Net, shrinking has been previously seen in Table \\ref{tab:test-data-first-results}",
    label="tab:test-results-data-augmentation"
)
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

eval_res.groupby(["Model"]).agg(['mean', 'std']).apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/eval-results-mean.tex",
    caption=
    "Performance results from various models that where optimized on from validation data, results shown in this table is from testing with the never before seen test dataset, also called evaluation dataset, it has only been run once in this run, and nothing was changed again afterwords",
    label="tab:eval-results"
)
eval_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/eval-results-full.tex",
    caption="Full results from the models when tested on the final evaluation set",
    label="tab:eval-results-full"
)

#%%
files = glob.glob("./logs/UNet*_*scaled-[012]-final-evaluate.csv")
files += glob.glob("./logs/UNet*_*normal-[012]-final-evaluate.csv")
files.sort()
scaled_res = pd.concat([pd.read_csv(i) for i in files]
                      ).drop(columns=["epoch", "fg_f1"]).sort_index()
idxs = [["3D U-Net, padding", "2D U-Net"], list(range(3))]
idxs = pd.MultiIndex.from_product(idxs, names=["Model", "i"])
scaled_res.index = idxs
scaled_res = scaled_res.rename(
    columns=lambda x: x.capitalize() if "fg_" not in x else x[3:].capitalize(),
)
scaled_res.applymap(round_vals).to_latex(
    **latex_kwargs,
    buf="./tables/scaled-results-full.tex",
    caption=
    "The full results from the models trained with Normalization and standardization",
    label="tab:scaled-results-full",
)
scaled_res.groupby(["Model"]).aggregate('mean').apply(bold_max).to_latex(
    **latex_kwargs,
    buf="./tables/scaled-results-mean.tex",
    caption=
    "The results from the models trained with Normalization and standardization",
    label="tab:scaled-results-mean",
)
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
dict_val["Total"] = counts
data = pd.DataFrame(dict_val).set_axis(["M", 'F']).T
data[["M\%", "F\%"]] = (data.iloc[:, :] / data.sum(axis=1)[:, None])
data = data.sort_index(axis=1)
data.to_latex(
    **latex_kwargs,
    buf="./tables/men-women-split.tex",
    caption=
    "The split-wise and total percentage of the data that is men vs women",
    label="tab:men-women-per-split"
)
# %%
