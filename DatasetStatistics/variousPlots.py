#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 22:00:14 2023

@author: ilanadeutsch, shenmena
"""

import os
import pydicom as dicom
import pandas as pd
import matplotlib as plt
from matplotlib import figure, pyplot
from statistics import median, mean
import numpy as np
import SimpleITK as sitk
import csv
import seaborn as sns
from tabulate import tabulate



# Set variables
m_yes = 0
m_no = 0
f_yes = 0
f_no = 0

y_age = []
n_age = []

y_pixel = []
n_pixel = []

y_slice = []
n_slice = []
y_scanner = []
n_scanner = []

y_kvp = []
n_kvp = []

# Load data
dataPath = "/home/ilanadeutsch/Desktop/TextureAnalysisFinal"
dataDir = os.listdir(dataPath)

# Load data curation
curation = pd.read_excel(r"/home/ilanadeutsch/Desktop/BAP1 data curation.xlsx", sheet_name = "Disease laterality - Feng")

for caseNum, case in enumerate(dataDir):

    # Skip invisible entry
    if case == ".DS_Store":
        continue

    # Define image path
    imgsPath = os.path.join(dataPath, dataDir[caseNum], "OriginalImgs")
    imgsDir = os.listdir(imgsPath)
    
    # Read in dicom info
    ds = dicom.read_file(os.path.join(imgsPath, imgsDir[0]))

    # Link to patient to BAP1 status
    for ipatient,patient in enumerate(curation["Case"]):
        if patient in case:
            status = curation.iloc[ipatient,7]
            
    if case == "135_18000101_CT_S_T_RECON_left" or case == "14_18000101_CT_STANDARD_AXIAL_left":
        status = "Yes"
        print(status)

    # Determine sex
    if ds.PatientSex == "M":
       if status == "Yes":
         m_yes = m_yes + 1
       else:
           m_no = m_no +1
    if ds.PatientSex == "F":
        if status == "Yes":
            f_yes = f_yes + 1
        else:
            f_no = f_no +1

    # Determine age
    if status == "Yes":
        y_age.append((int(ds.PatientAge[1:3])))
    else:
        n_age.append((int(ds.PatientAge[1:3])))

    # Scanner data
    if status == "Yes":
        y_slice.append(int(ds.SliceThickness))
        y_scanner.append(ds.Manufacturer)
        y_pixel.append(float(ds.PixelSpacing[0]))
        y_kvp.append(float(ds.KVP))
    else:
        n_slice.append(int(ds.SliceThickness))
        n_scanner.append(ds.Manufacturer)
        n_pixel.append(float(ds.PixelSpacing[0]))
        n_kvp.append(float(ds.KVP))
        
# Calculate medians and ranges
y_median = median(y_age)
n_median = median(n_age)

y_range = f"{min(y_age)}-{max(y_age)}"
n_range = f"{min(n_age)}-{max(n_age)}"

t_age = y_age + n_age
t_median = int(median(t_age))
t_range = f"{min(t_age)}-{max(t_age)}"

y_slice_median = median(y_slice)
y_slice_range = f"{min(y_slice)}-{max(y_slice)}"
n_slice_median = median(n_slice)
n_slice_range = f"{min(n_slice)}-{max(n_slice)}"

t_slice = y_slice + n_slice
t_slice_median = median(t_slice)
t_slice_range = f"{min(t_slice)}-{max(t_slice)}"

# Sums
n_GE = len([scanner for scanner in n_scanner if scanner == "GE MEDICAL SYSTEMS"])
y_GE = len([scanner for scanner in y_scanner if scanner == "GE MEDICAL SYSTEMS"])

n_philips = len([scanner for scanner in n_scanner if scanner == "Philips"])
y_philips = len([scanner for scanner in y_scanner if scanner == "Philips"])

n_toshiba = len([scanner for scanner in n_scanner if scanner == "TOSHIBA"])
y_toshiba = len([scanner for scanner in y_scanner if scanner == "TOSHIBA"])

n_siemens = len([scanner for scanner in n_scanner if scanner == "SIEMENS"])
y_siemens = len([scanner for scanner in y_scanner if scanner == "SIEMENS"])

y_pixel_median = "{:.2}".format(median(y_pixel))
y_pixel_range = f"{round(min(y_pixel),2)}-{round(max(y_pixel),2)}"

n_pixel_median = "{:.2f}".format(median(n_pixel))
n_pixel_range = f"{round(min(n_pixel),2)}-{round(max(n_pixel),2)}"

t_pixel_median = "{:.2f}".format(median(n_pixel + y_pixel))
t_pixel_range = f"{round(min(n_pixel + y_pixel),2)}-{round(max(n_pixel + y_pixel),2)}"

n_kvp_median = "{:.2f}".format(median(n_kvp))
n_kvp_range = f"{round(min(n_kvp),2)}-{round(max(n_kvp),2)}"

y_kvp_median = "{:.2f}".format(median(y_kvp))
y_kvp_range = f"{round(min(y_kvp),2)}-{round(max(y_kvp),2)}"

t_kvp_median = "{:.2f}".format(median(n_kvp+y_kvp))
t_kvp_range = f"{round(min(n_kvp),2)}-{round(max(n_kvp+y_kvp),2)}"

# Display Demographic Data
headers = ["Characteristic","Total (n = 131)", "BAP1 Mutation Status [+] (n = 60) ","BAP1 Mutation Status [-] (n = 71)"]
data = [["Sex","","",""],["Male", m_yes + m_no ,m_yes,m_no],["Female",f_yes + f_no ,f_yes,f_no],["Age","","",""],["Median",int(t_median),y_median,n_median],["Range",t_range,y_range,n_range]]

print(tabulate(data, headers=headers))

# CT data
headers = ["Characteristic","Total (n = 131)", "BAP1 Mutation Status [+] (n = 60) ","BAP1 Mutation Status [-] (n = 71)"]
data = [["Pixel Size [mm]","","",""],["Median",t_pixel_median,y_pixel_median,n_pixel_median], ["Range",t_pixel_range,y_pixel_range,n_pixel_range],
        ["Slice Thickness [mm]","","",""],["Median","3",y_slice_median,n_slice_median],["Range","1-5",y_slice_range,n_slice_range],["kVp [kV]","","",""],
        ["Median",t_kvp_median,y_kvp_median,n_kvp_median],["Range",t_kvp_range,y_kvp_range,n_kvp_range],["Scanner Manufacturer","","",""],
        ["GE",y_GE + n_GE,y_GE,n_GE],["Philips",y_philips + n_philips,y_philips,n_philips],
        ["Toshiba", y_toshiba+n_toshiba,y_toshiba,n_toshiba],["Siemens",y_siemens +n_siemens,y_siemens,n_siemens],]

print(tabulate(data, headers=headers))

#%% Initialize variables
size = []
names = []

y_size = []
n_size = []

tot_y_size = []
tot_n_size = []

y_size_print = []
n_size_print = []

for caseNum, case in enumerate(dataDir):

    # Reset total area for new case
    tot_size_temp = 0

    # Skip invisible entries
    if case == ".DS_Store":
        continue

    # Save a list of the case names
    names.append(case)

    # Define image path
    imgsPath = os.path.join(dataPath, dataDir[caseNum], "OriginalImgs")
    imgsDir = os.listdir(imgsPath)
        
    # Read in dicom info
    ds = dicom.read_file(os.path.join(imgsPath, imgsDir[0]))

    # Define map
    mapsPath = os.path.join(dataPath, dataDir[caseNum], "Masks_Resampled")
    mapsDir = os.listdir(mapsPath)

    # Loop through slices
    for sliceNum, slice in enumerate(mapsDir):

        # Define probability map
        mapPath = os.path.join(mapsPath, mapsDir[int(sliceNum)])
        map = sitk.ReadImage(mapPath)
        map = sitk.GetArrayFromImage(map)
        map[map==255] = 1

        # Link to patient to BAP1 status
        for ipatient,patient in enumerate(curation["Case"]):
            if patient in case:
                status = curation.iloc[ipatient,7]
                
                    
        if case == "135_18000101_CT_S_T_RECON_left" or case == "14_18000101_CT_STANDARD_AXIAL_left":
            status = "Yes"
            print(status)
                
        # Save tumor size according to BAP1 status
        if status == "Yes":
            if sliceNum == 1:
                y_size.append(sum(map[map==1])*(ds.PixelSpacing[0]*ds.PixelSpacing[1]))
        else:
            if sliceNum == 1:
                n_size.append(sum(map[map==1])*(ds.PixelSpacing[0]*ds.PixelSpacing[1]))

        tot_size_temp = + tot_size_temp + sum(map[map==1])*ds.PixelSpacing[0]*ds.PixelSpacing[1]

    # Save cumulative tumor size
    if status == "Yes":
        tot_y_size.append(tot_size_temp)
    else:
        tot_n_size.append(tot_size_temp)

# Multiple by 3 to find volume
tot_y_size = [size*3 for size in tot_y_size]
tot_n_size = [size*3 for size in tot_n_size]

# Plot histogram of tumor area for middle slice

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

plt.pyplot.xlabel("Area [$mm^2$]")
plt.pyplot.title("Tumor Area of the Representative Slice")
plt.pyplot.ylabel("Number of Cases")
sns.set_theme(style='darkgrid')
plt.pyplot.hist([y_size, n_size],color=['r','b'], alpha=0.5, label=['BAP1 [+]','BAP1 [-]'] )
plt.pyplot.legend()
plt.pyplot.ticklabel_format(style='plain')

# Plot tumor across 3 slices
plt.pyplot.hist([tot_y_size, tot_n_size],color=['r','b'], alpha=0.5, label=['BAP1 [+]','BAP1 [-]'] )
plt.pyplot.legend()
plt.pyplot.title("Tumor Volume")
plt.pyplot.xlabel("Volume [$mm^3$]")
plt.pyplot.ylabel("Number of Cases")

# Plot age histogram
plt.pyplot.xlabel("Age")
plt.pyplot.ylabel("Number of Cases")
plt.pyplot.title("Patient Age Distribution")
plt.pyplot.hist([y_age,n_age], color=['r','b'], alpha=0.5,label=['BAP1 [+]','BAP1 [-]'])
plt.pyplot.vlines(x = 69 , ymin = 0, ymax = 25,
          linestyles = "dashed", colors = "black", label = "Median = 69")  
plt.pyplot.legend()

import scipy

print(f"Mean BAP1 [-] size: {mean(n_size)}")
print(f"Mean BAP1 [+] size: {mean(y_size)}")

print(f"Median BAP1 [-] size: {median(n_size)}")
print(f"Median BAP1 [+] size: {median(y_size)}")

# For differences in tumor area
scipy.stats.ranksums(n_size,y_size)

# For differences in age
scipy.stats.ranksums(n_age, y_age)

# For differences in tumor volume
scipy.stats.ranksums(tot_n_size, tot_y_size)