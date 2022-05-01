from __future__ import absolute_import
from __future__ import print_function

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from mimic3models.in_hospital_mortality.helper import InHospitalMortalityReader, Discretizer, Normalizer, \
    read_and_extract_features, read_chunk

import os
import numpy as np
import argparse
import json
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='all', help='specifies what features to extract',
                        choices=['all', 'len', 'all_but_len'])
    parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                        default=os.path.join(os.path.dirname(__file__), 'data/'))
    parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                        default='.')
    parser.add_argument('--period', type=str, default='all', help='specifies which period extract features from',
                        choices=['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all'])
    args = parser.parse_args()

    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                             listfile=os.path.join(args.data, 'train_listfile.csv'),
                                             period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                           listfile=os.path.join(args.data, 'val_listfile.csv'),
                                           period_length=48.0)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                            listfile=os.path.join(args.data, 'test_listfile.csv'),
                                            period_length=48.0)

    print('Reading data and extracting features ...')
    (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
    (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
    (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
    print('train data shape', train_X.shape)
    print('validation data shape', val_X.shape)
    print('test data shape', test_X.shape)

    print('Imputing missing values ...')
    imputer = SimpleImputer(missing_values=np.NAN, strategy='mean', fill_value=None, verbose=0, copy=True)
    imputer.fit(train_X)
    train_X = np.array(imputer.transform(train_X), dtype=np.float32)
    val_X = np.array(imputer.transform(val_X), dtype=np.float32)
    test_X = np.array(imputer.transform(test_X), dtype=np.float32)

    print('Normalizing the data to have zero mean and unit variance ...')
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    print("Reading values into pandas...")
    import pandas as pd
    trainX = pd.DataFrame(train_X)
    trainY = pd.DataFrame(train_y)
    valX = pd.DataFrame(val_X)
    valY = pd.DataFrame(val_y)
    testX = pd.DataFrame(test_X)
    testY = pd.DataFrame(test_y)

    print("Reading Names into csv")
    pd.DataFrame(train_names).to_csv("trainLG_names.csv")
    pd.DataFrame(val_names).to_csv("valLG_names.csv")
    pd.DataFrame(test_names).to_csv("testLG_names.csv")

    print('Exporting to csv')
    trainX.to_csv('trainX.csv')
    trainY.to_csv('trainY.csv')

    valX.to_csv('valX.csv')
    valY.to_csv('valY.csv')

    testX.to_csv('testX.csv')
    testY.to_csv('testY.csv')


if __name__ == '__main__':
    main()
