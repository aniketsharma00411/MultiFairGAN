# MultiFairGAN
This repository contains the code for the paper [MultiFairGAN: Achieving Fairness in Synthetic Data for Clinical Records across
Multiple Protected Attributes](#).

## Table of Contents
- [Abstract](#abstract)
- [Generating Fair Synthetic Data](#generating-fair-synthetic-data)
- [Running Experiments](#running-experiments)
- [Citing](#citing)

## Abstract
Utilizing the true potential of Machine Learning in transforming healthcare positively demands access to Electronic Health Records (EHR). However, these records contain sensitive patient information. To ensure patient privacy, synthetic data is used as an alternative to the real data. Often, clinical datasets have an inequitable representation of various subgroups based on protected attributes such as race, gender, insurance, and age which gets mirrored or even exacerbated in the synthetic datasets. Such imbalanced representation poses a threat of biased decisions towards minority subgroups leading to adverse consequences and less healthcare access. Addressing the need for fair synthetic data in healthcare, we perform a case study on the possibility of generating fair data by modifying the adversarial objective of the generator to include a fairness constraint. We propose MultiFairGAN which uses the Demographic Parity Ratio (DPR) and the Demographic Parity Difference (DPD) scores for fairness to generate synthetic data. We study the effectiveness of the proposed setting on two clinical datasets, MIMIC-III and eICU. Contrasting other works which assume binary protected attributes for fairness, MultiFairGAN calculates fairness constraint values with respect to more than one non-binary protected attributes considered together. Results indicate that MultiFairGAN generates data representing diverse minority subgroups more adequately than the other baseline models for fair synthetic data generation and demonstrates improved fairness without compromising much on the utility.

## Generating Fair Synthetic Data
To train the MultiFairGAN model and generate data use the below command.
```bash
python run_multifairgan.py --file_name <file-name>.csv --dataset <dataset>
```

Here, <dataset> can either be MIMIC or EICU.

Similarly, [TabFairGAN](https://doi.org/10.3390/make4020022) and [Distance Correlation GAN](https://doi.org/10.1007/978-3-031-35891-3_26) models can also be used.

The parameter settings for each of these models can be set in their respective Python files.

## Running Experiments
The [Experiment Notebooks](Experiment%20Notebooks) directory contain the Google Colab notebooks for each of the experiments performed with the results.

## Citing
If you use the code or any part of it, please cite the paper:
```
```
