#!/usr/bin/env python3
import subprocess
import os
os.chdir("/homes/pmcd/Peter_Patrick3/")
subprocess.call("mp train --overwrite", shell=True)
subprocess.call("mp train_fusion --overwrite", shell=True)
subprocess.call("mp predict --overwrite", shell=True)