# Multi-Input Transformer for EEG-based Classification

This repository contains the implementation of a multi-input transformer model for EEG-based classification. The model leverages self-attention and cross-attention mechanisms to process and classify EEG data.

## Overview

The code is designed to train and evaluate a multi-input transformer model on EEG data. The model is capable of handling multiple input channels and utilizes various neural network components, including self-attention, cross-attention, and transformer blocks.

## Data

The dataset consists of EEG recordings from 30 subjects. The experiment paradigm follows the structure described in the referenced papers with specific details as follows:

- **Subjects 1-20:** The experiment paradigm is based on the original design described in the related papers.
- **Subjects 21-30:** The rest period between sessions has been modified from 180 seconds to 210 seconds to investigate the effect of a longer rest period on EEG signals.

## Repository Structure

- `30/`: Directory containing the original EDF files for the 30 subjects.
- `特征/`: Directory containing the extracted features from the EEG data, already divided into training and testing sets.
- `SAPR-Net.py`: Script containing the implementation of the multi-input transformer model, including data loading, training, and evaluation functions.

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
