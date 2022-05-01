# 707-data-processing

Our group utilized helper codes from the original paper to preprocess the hospital mortality data. Functions that were adapted from the original paper is in the helper.py file. For data used for deep learning models, they were initially processed into 2D data to be exported and can be reconverted with the command ```3D_file = 2D_file.reshape(2D_file.shape[0], 2D_file.shape[1] // 76, 76)```

To preprocess data for Deep Learning Models use: 

``` python -um mimic3models.in_hospital_mortality.main ```

To preprocess data for Logistic Regression use:Cancel changes

```python -um mimic3models.in_hospital_mortality.logistic_main```

Link to Google Colab for building our models
https://colab.research.google.com/drive/127Umq4NpOgvJvIEDmH7EwWSPD2uXBqt8?authuser=2#scrollTo=dRDh-yIRHXbi
