#%%
import glob
from sys import exec_prefix
import dicom2nifti.convert_dir as conv_dir
import os
import shutil
import json
os.chdir("/homes/pmcd/Peter_Patrick3")
base_dir = "/homes/pmcd/Peter_Patrick3"

#%%
folders = glob.glob("/homes/pmcd/Peter_Patrick3/DICOM/*/*")
try:
    os.mkdir("/homes/pmcd/Peter_Patrick3/all")
except:
    pass
for idx, folder in enumerate(folders):
    file_name = f"anon{idx}"
    print(file_name)
    out_dir = base_dir + "/all/" + file_name
    nii_file = glob.glob(f"{folder}/*.nii.gz")
    completed = {
        "rest": False,
        "stress": False,
        "ct": False,
        "original": folder
    }
    anon_file = f"{out_dir}/annotations.nii.gz"
    try:
        os.mkdir(out_dir)
        shutil.copy(nii_file[0], anon_file)
    except:
        pass
    conv_dir.convert_directory(folder, out_dir, True, False)
    for i in glob.glob(f"{out_dir}/*"):
        if "rest" in i.lower():
            os.rename(i, f"{out_dir}/rest.nii.gz")
            completed["rest"] = True
        elif "stress" in i.lower():
            os.rename(i, f"{out_dir}/stress.nii.gz")
            completed["stress"] = True
        elif "ct" in i.lower():
            os.rename(i, f"{out_dir}/ct.nii.gz")
            completed["ct"] = True

    with open(f'{out_dir}/data.json', 'w') as outfile:
        json.dump(completed, outfile)
    # break
#%%
train_split = 0.8
chances = {
    "train": train_split * 0.6,
    "val": train_split * 0.2,
    "test": train_split * 0.2,
    "true_test": 1 - train_split
}
#%%
examples = [
    folder for folder in glob.glob(f"{base_dir}/all/*")
    if os.path.exists(f"{folder}/rest.nii.gz") and
    os.path.exists(f"{folder}/annotations.nii.gz")
]
data = {}
sum_val = 0
for key, value in chances.items():
    end = sum_val + (len(examples) * value)
    first_idx = int(sum_val)
    end_idx = round(end)
    print(end, end_idx, first_idx, sum_val)
    sum_val = end_idx
    data[key] = examples[first_idx:end_idx]
data[key] = examples[first_idx:]
print(len([j for i in data.values() for j in i]))
#%%
idx = 0
for ds_name, folders in data.items():
    ds_base_folder = f"/homes/pmcd/Peter_Patrick3/{ds_name}"
    try:
        os.mkdir(ds_base_folder)
    except:
        pass

    for folder in folders:
        file_name = folder.split("/")[-1]
        try:
            os.mkdir(f"{ds_base_folder}/images")
            os.mkdir(f"{ds_base_folder}/labels")
        except:
            pass
        if os.path.exists(f"{folder}/rest.nii.gz"
                         ) and os.path.exists(f"{folder}/annotations.nii.gz"):
            os.symlink(
                f"{folder}/rest.nii.gz",
                f"{base_dir}/{ds_name}/images/{file_name}.nii.gz"
            )
            os.symlink(
                f"{folder}/annotations.nii.gz",
                f"{base_dir}/{ds_name}/labels/{file_name}.nii.gz"
            )
