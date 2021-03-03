#%%
import glob
import dicom2nifti.convert_dir as conv_dir
import os
import shutil
os.chdir("/homes/pmcd/Peter_Patrick3")

#%%
folders = glob.glob("/homes/pmcd/Peter_Patrick3/DICOM/*/*")
for idx,folder in enumerate(folders):
  out_dir = f"/homes/pmcd/Peter_Patrick3/res/anon{idx}"
  try:
    os.mkdir(out_dir)
  except:
    pass
  conv_dir.convert_directory(folder, out_dir, False, False)
  for i in glob.glob(f"{out_dir}/*"):
    if "rest" in i.lower():
      os.rename(i, f"{out_dir}/rest.nii")
    elif "stress" in i.lower():
      os.rename(i, f"{out_dir}/stress.nii")
    else:
      os.rename(i, f"{out_dir}/ct.nii")
        
#%%
folders = glob.glob("/homes/pmcd/Peter_Patrick3/res/*")
remove_dirs = []
for i in folders:
  if len(glob.glob(f"{i}/*")) != 3:
    remove_dirs.append(i)
for i in remove_dirs:
  shutil.rmtree(i)
#%%
folders = glob.glob("/homes/pmcd/Peter_Patrick3/res/*")
for idx,folder in enumerate(folders):
  new_name = f"/homes/pmcd/Peter_Patrick3/res/anon{idx}"
  os.rename(folder, new_name)