#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:37:46 2021

@author: mmolina
"""
# Installs RoI align extension
import os, sys, subprocess

def pip_install(item):
    subprocess.check_call([sys.executable, "-m", "pip", "install", item])

def install_custom_ext(setup_path):
    try:
        pip_install(setup_path)
    except Exception as e:
        print("Could not install custom extension {} from source due to Error:\n{}\n".format(path, e) +
              "Trying to install from pre-compiled wheel.")
        dist_path = setup_path+"/dist"
        wheel_file = [fn for fn in os.listdir(dist_path) if fn.endswith(".whl")][0]
        pip_install(os.path.join(dist_path, wheel_file))

custom_exts =  ["custom_extensions/roi_align/2D", "custom_extensions/roi_align/3D"]
for path in custom_exts:
    try:
        install_custom_ext(path)
    except Exception as e:
        print("FAILED to install custom extension {} due to Error:\n{}".format(path, e))