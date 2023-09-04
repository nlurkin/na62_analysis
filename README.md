# UCLouvain CP3 NA62 analysis

This project provides tools and exercises to perform data analysis on a simplified data sample from NA62.
A series of jupyter notebooks are available. When looked at in order, they will provide a gentle introduction to the available data and to
several common analysis techniques used in NA62 before finishing with a complete analysis.

Some python tools are available in the na62 package to help with some tasks that are either specific to the data format provided, or specific to NA62 and not necessarily common knowledge.

## Installation - Requirements
It is recommended to run the code in this repository inside a dedicated conda environment.

If it is not yet available on your system, please first [install miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/index.html).
Alternatively, if you really don't want or really can't use a conda installation, the list of python packages that you will need can be found in the [environment.yaml](environment.yml) file.

The repository includes a large datafile to run examples without access to the complete dataset. Obtaining this file from the repository requires the package `git-lfs` to be installed on your system. Make sure it is available before cloning the repository, or the example ROOT data file will only contain a short text link. Don't forget to `git lfs install`.

Then download the repository, create and activate the conda environment:
```
git clone https://cp3-git.irmp.ucl.ac.be/na62/na62_analysis.git
cd na62_analysis
conda env create -f environment.yml
conda activate ucl
```

This last step (activating the environment) needs to be done in every new terminal that you will be opening.

## Usage
To run the notebooks, please go inside the repository, activate the environment and run
```
jupyter-lab "00 - START HERE.ipynb" # This automatically starts the first notebook
```

You will be redirected to the interactive web application. In the menu on the left, you are now free to select the notebook that you want to run.
Please refer to the jupyter documentation for more information on how to use jupyter(-lab) and notebooks.
Otherwise, you may now follow the instructions in the different notebooks.


# Code documentation
The latest version of the complete documentation of the code is automatically compiled. Please run the `download_doc.sh`
script to download the latest version. Then open the *doc_compiled/index.html* file in your web browser or run
```
firefox doc_compiled/index.html &
```
to browse the documentation (or use your favourite browser).


# Data files
If you have configured git-lfs correctly, you should already have a limited amount of data available in the **data** directory. This is enough for most of the exercises. They contain the data from one full run (12450, chosen at random) and a small MC simulated sample of each of the five main kaon decay modes (about 50 MB worth of events for each channel).

In some of the exercises, and the final one, more statistics than what is available here is required. You will find a more complete sample of data on the [sharepoint of the course](https://uclouvain.sharepoint.com/:f:/s/O365G-LPHYS2233_NA62/ElCQK-vssOxAj-m9S7KcvsQBcpuWZ1Eqq7LliWjW0p4PeA?e=RxtlA3). For those exercises, please download the **Full_DataSample** and place it in the root directory of this project, in a directory named **Full_DataSample**.

*WARNING*: The full data sample is about 21 GB in size. Make sure you have a reliable connection while downloading. In case of troubles, consider downloading each subdirectory separately.

# TODO
 - [ ] Add theoretical propagation through spectrometer magnet
 - [ ] Possible updates in selections. May reduce too much the acceptance ?
  - [ ] Add more timing cuts: MUV3 done, but KTAG/Event can be done (ktag-event, or track-ktag or track-trigger or track-vertex)
  - [ ] Add pile-up constraints ? In single track events, no other track within 10 ns?
  - [ ] Add constraint on reco Pi0
  - [ ] Add more constraints on track momentum ? (done in some pre-selection but not everywhere and not in current selections)
 - [ ] Make sure that the cut values are explained (best EoP cuts, best timing cuts, ...)
 - [ ] Some issue whith running large data statistics: plots look completely off by orders of magnitude. Suspect issue with one or more data files (issue shows when running on all data files with "small" MC stats, but not when running only chunks of the data files).