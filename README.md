# Mayo Sleep Spindles Research Project

## Project Overview
This repository contains the implementation and research work related to sleep spindle detection and analysis conducted at Mayo Clinic. Sleep spindles are thalamocortical oscillations occurring during non-REM sleep that play crucial roles in memory consolidation, cognition, and neural development.

## Graphical Abstract
![Graphical Abstract: Sleep Spindles](graphical_abstract_spindles.pdf)

## Scalp and iEEG Data Visualization
![Scalp and Intracranial EEG Data](scalp_ieeg_data.pdf)

## Research Highlights
- Advanced detection of sleep spindles using neural network approaches
- Analysis of spindle characteristics across different patient populations
- Integration of scalp and intracranial EEG data for comprehensive spindle characterization

### Installation Guide
On Linux and Windows the project can be used by running the following commands:

#### Using `pip` with virtualenv for Windows
```shell
# enter the project root
git clone https://github.com/yourusername/mayo_spindles.git
cd mayo_spindles
# Create the virtual env with pip
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# With jupyter notebook
pip install ipykernel
pip install ipywidgets
```


## Directory Structure
- `mayo_spindles/`: Core implementation of the project
- `model_repo/`: Repository of various models used in the project

