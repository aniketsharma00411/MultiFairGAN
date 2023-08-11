from distcorrgan.distcorrgan import train_plot, get_original_data
import argparse
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_name", help="name of the produced csv file", type=str)
parser.add_argument(
    "--dataset", help="name of the dataset to use (MIMIC/EICU)", type=str)
args = parser.parse_args()

if args.dataset == "MIMIC":
    # MIMIC
    S = 'ethnicity'
    Y = 'hospital_expire_flag'
    S_under = 'African American'
    Y_desire = '0'
elif args.dataset == "EICU":
    # EICU
    S = 'ethnicity'
    Y = 'hospitaldischargestatus'
    S_under = 'NON-WHITE'
    Y_desire = '0'
else:
    raise ValueError(
        "Wrong dataset value. Please choose either MIMIC or EICU.")

if args.dataset == "MIMIC":
    # MIMIC
    data = pd.read_csv('MIMIC_TAB.csv')
elif args.dataset == "EICU":
    # EICU
    data = pd.read_csv('eicu_age.csv')
else:
    raise ValueError(
        "Wrong dataset value. Please choose either MIMIC or EICU.")

data[S] = data[S].astype(object)
data[Y] = data[Y].astype(object)

print("done")
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and 1 > 0) else "cpu")

generator, critic, ohe, scaler, data_train, data_test, input_dim = train_plot(
    data,
    S,
    Y,
    S_under,
    Y_desire,
    2,
    0.2
)
print("done")

# Synthetic copy
if args.dataset == "MIMIC":
    # MIMIC
    fake_numpy_array = generator(torch.randn(
        size=(8772, input_dim), device=device)).cpu().detach().numpy()

elif args.dataset == "EICU":
    # EICU
    # fake_numpy_array = generator(torch.randn(
    #     size=(53265, input_dim), device=device)).cpu().detach().numpy()
    fake_numpy_array = generator(torch.randn(
        size=(50978, input_dim), device=device)).cpu().detach().numpy()


fake_df = get_original_data(fake_numpy_array, data, ohe, scaler)
fake_df = fake_df[data.columns]
fake_df.to_csv(args.file_name)
