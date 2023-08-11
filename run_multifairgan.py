from multifairgan.multifairgan import MultiFairGANSynthesizer
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_name", help="name of the produced csv file", type=str)
parser.add_argument(
    "--dataset", help="name of the dataset to use (MIMIC/EICU)", type=str)
args = parser.parse_args()

# Names of the columns that are discrete
if args.dataset == "MIMIC":
    # MIMIC
    columns = [
        'gender', 'ethnicity',
        'insurance', 'diagnosis_at_admission', 'discharge_location',
        'fullcode_first', 'dnr_first', 'fullcode', 'dnr', 'cmo_first',
        'cmo_last', 'cmo', 'los_icu', 'admission_type', 'hospital_expire_flag',
        'readmission_30', 'max_hours']

    sensitive_attr = ['ethnicity', 'insurance']
    # sensitive_attr = ['ethnicity']
    label = 'hospital_expire_flag'
elif args.dataset == "EICU":
    # EICU
    columns = ['gender', 'ethnicity', 'hospitalid',
               'wardid', 'apacheadmissiondx', 'admissionheight', 'hospitaladmitsource',
               'hospitaldischargeyear', 'hospitaldischargestatus', 'unittype',
               'admissionweight', 'dischargeweight', 'AgeGroup']

    sensitive_attr = ['ethnicity', 'AgeGroup']
    # sensitive_attr = ['ethnicity']
    label = 'hospitaldischargestatus'
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

multifairgan = MultiFairGANSynthesizer(
    epochs=200,
    fairness_loss='dpr',
    fairness_epochs=20,
    fairness_coef=0.2,
    additive_loss=False
)

print("done")
multifairgan.fit(data, sensitive_attr, label, columns)
print("done")

# Synthetic copy
if args.dataset == "MIMIC":
    # MIMIC
    samples = multifairgan.sample(8772)
elif args.dataset == "EICU":
    # EICU
    # samples = multifairgan.sample(53265)
    samples = multifairgan.sample(50978)

samples.to_csv(args.file_name)
