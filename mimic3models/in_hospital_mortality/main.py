from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import pandas as pd
import argparse
from mimic3models.in_hospital_mortality.helper import InHospitalMortalityReader, Discretizer, Normalizer, load_data


parser = argparse.ArgumentParser()
parser.add_argument('--timestep', type=float, default=1.0, help="fixed timestep used in the dataset")
parser.add_argument('--normalizer_state', type=str, default=None,
                    help='Path to a state file of a normalizer. Leave none if you want to '
                         'use one of the provided ones.')
parser.add_argument('--imputation', type=str, default='previous')
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default=os.path.join(os.path.dirname(__file__), 'data/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--small_part', dest='small_part', action='store_true')
args = parser.parse_args()


# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

discretizer = Discretizer(timestep=float(args.timestep),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'ihm_ts{}.input_str:{}.start_time:zero.normalizer'.format(args.timestep, args.imputation)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)


# Reading data
print("Reading Train Data...")
train_raw = load_data(train_reader, discretizer, normalizer, args.small_part)
print("Successfully loaded!")

print("Reshape Train Files...")
trainX_raw_2d = np.array(train_raw[0]).reshape(np.array(train_raw[0]).shape[0], -1)

print("Train Files Shape...")
print("train X shape", np.array(train_raw[0]).shape)
print("train 2d shape", np.array(trainX_raw_2d).shape)
print("train Y shape", np.array(train_raw[1]).shape)

print("Saving Train Files...")
np.savetxt('trainX_DL.txt', trainX_raw_2d)
np.savetxt('trainY_DL.txt', train_raw[1])

# Reading Val Data
print("Reading Val Data...")
val_raw = load_data(val_reader, discretizer, normalizer, args.small_part)
print("Successfully loaded!")

print("Reshape Val Files...")
valX_raw_2d = np.array(val_raw[0]).reshape(np.array(val_raw[0]).shape[0], -1)

print("Val Files Shape...")
print("val X shape", np.array(val_raw[0]).shape)
print("val 2d shape", valX_raw_2d.shape)
print("val Y shape", np.array(val_raw[1]).shape)

print("Saving Validation Files...")
np.savetxt('valX_DL.txt', valX_raw_2d)
np.savetxt('valY_DL.txt', val_raw[1])

print("Reading Test Data...")
# ensure that the code uses test_reader
del train_reader
del val_reader
del train_raw
del val_raw

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)
ret = load_data(test_reader, discretizer, normalizer, args.small_part,
                      return_names=True)

test_raw = load_data(test_reader, discretizer, normalizer, args.small_part,
                return_names=True)

data = np.array(ret["data"][0])
labels = np.array(ret["data"][1])
names = np.array(ret["names"])
print("Successfully loaded!")

print("Reshape Test Files...")
testX_raw_2d = data.reshape(data.shape[0], -1)

print("test X shape", np.array(data).shape)
print("test 2D shape", np.array(testX_raw_2d).shape)
print("test Y shape", labels.shape)
pd.DataFrame(names).to_csv("names_CNN.csv")

print("Saving Test Files...")
np.savetxt('testX_DL.txt', testX_raw_2d)
np.savetxt('testY_DL.txt', labels)
print("Successfully processed all files!")