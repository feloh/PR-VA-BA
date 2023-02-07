from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
from pyspin.spin import make_spin, Default


# Loading Function, Spliting the Data into Test and Train Data, Using SMOTE and RandomUndersampler to balance the Data
@make_spin(Default, "Loading the Dataset...")
def load_data(path, s, test_split, smote, under_sampler):
    print('Path: ', path)
    ds = np.loadtxt('data/Output.csv', delimiter=',')
    if test_split:
        split = int(len(ds) * 0.3)
        ds = ds[0:split]
    seed = s
    np.random.seed(seed)
    input = ds[:, 0:39]
    output = ds[:, 39]
    x_train, x_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=seed)

    # define pipeline
    over = SMOTE(sampling_strategy=smote)
    under = RandomUnderSampler(sampling_strategy=under_sampler)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    x_train, y_train = pipeline.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test
