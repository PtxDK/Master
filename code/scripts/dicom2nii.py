#%%
from pydicom import dcmread
import glob
from multiprocessing.pool import Pool

#%%
folders = glob.glob("/homes/pmcd/Peter_Patrick3/DICOM/*/*")
# for folder in folders:
def get_info(folder):
  vals = map(dcmread, glob.glob(f"{folder}/*"))
  v = {"REST": [], "STRESS": [], "CT": []}
  for val in (dcmread(file) for file in glob.glob(f"{folder}/*") if "." not in file):
      key = [k for k in v.keys() if k in val[(0x0008, 0x103e)].value]
      v[key[0]].append(val)
  return v
res = map(get_info, folders)
#%%
