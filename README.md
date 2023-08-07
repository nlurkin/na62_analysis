# UCLouvain CP3 NA62 analysis

This project provides tools and exercises to perform data analysis on a simplified data sample from NA62.
A series of jupyter notebooks are available. When looked at in order, they will provide a gentle introduction to the available data and to
several common analysis techniques used in NA62 before finishing with a complete analysis.

Some python tools are available in the na62 package to help with some tasks that are either specific to the data format provided, or specific to NA62 and not necessarily common knowledge.

## Installation - Requirements
It is recommended to run the code in this repository inside a dedicated conda environment.

If it is not yet available on your system, please first [install miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/index.html).
Alternatively, if you really don't want or really can't use a conda installation, the list of python packages that you will need can be found in the [environment.yaml](environment.yml) file.

The repository includes a large datafile to run examples without access to the complete dataset. Obtaining this file from the repository requires the package `git-lsf1` to be installed on your system. Make sure it is available before cloning the repository, or the example ROOT data file will only contain a short text link.

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


# TODO
 - [ ] See if I can provide a documentation for the package (list of available functions)
 - [ ] Need to provide a way to access the data
