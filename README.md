# BOOLS database and SAPR-Net for brainprint recognition

This repository contains the implementation of the SAPR-Net and BOOLS database.

## Overview

The code is designed to train and evaluate a modified retention network based on the BOOLS database.

## Data

The dataset consists of EEG recordings from 30 subjects under resting, low, medium, and high cognitive load levels.

## Repository Structure

- `30/`: Directory containing the original EDF files for the 30 subjects.
- `特征/`: Directory containing the extracted features from the EEG data, already divided into training and testing sets.
- `SAPR-Net.py`: Brainprint recognizer

## Requirements

- Python 3.7 or higher
- NumPy
- Pandas
- Scikit-learn
- PyTorch 1.7.0 or higher
- Matplotlib
- Seaborn

Install the required packages using `pip`:

```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn
