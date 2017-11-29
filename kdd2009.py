import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.decomposition import PCA

def log():
    logging.basicConfig()
    logger = logging.getLogger('KDD2009')
    logger.setLevel(logging.INFO)
    return logger

logger = log()

############################## Read Dataset #################################################
logger.info("Start to fetch data set from local disk")

path = '/Users/gh603/Desktop/Github_Hui_Ge/KDD2009/orange_small_train.data'
train_path = path + '/orange_small_train.csv'
upselling_path = path + '/orange_small_train_upselling.labels'
test_path = path + '/orange_small_test.data'

X = pd.read_table(train_path)

Y = pd.read_table(upselling_path, header=None).ix[:, 0].astype('category')
Y.cat.rename_categories(['no', 'yes'], inplace=True)
print(Y.value_counts())

# convert numerical variable to float type and categorical variables to category type
X_colnames = X.columns
columnTypes = {X_colname: X[X_colname].dtype for X_colname in X_colnames}
for colname in X_colnames:
    col = X[colname]
    if columnTypes[colname] == int:
        col = col.astype(float)
    elif columnTypes[colname] != float:
        col = col.astype('category')
    X.ix[:, colname] = col
    columnTypes[colname] = col.dtype
logger.info('Finished fetching data set from local disk')

############################## Split Dataset #################################################
logger.info('Start to split data set into train set, validation set and test set randomly')

# Split X to train, validation and test
RANDOM_SEED = 1
train_proportion = 0.6
valid_proportion = 0.2

def train_validate_test_split(X, Y, train_percent, validate_percent, seed):
    np.random.seed(seed)
    perm = np.random.permutation(X.index)
    totLen = len(X)
    train_end = int(totLen * train_percent)
    validate_end = train_end + int(validate_percent * totLen)
    trainX = X.ix[perm[:train_end]]
    trainY = Y.ix[perm[:train_end]]
    validateX = X.ix[perm[train_end:validate_end]]
    validateY = Y.ix[perm[train_end:validate_end]]
    testX = X.ix[perm[validate_end:]]
    testY = Y.ix[perm[validate_end:]]
    return trainX, trainY, validateX, validateY, testX, testY

trainX, trainY, validateX, validateY, testX, testY \
    = train_validate_test_split(X, Y, train_proportion, valid_proportion, RANDOM_SEED)

# double check the dataset has been split representatively
Ys = dict(train=trainY, valid=validateY, test=testY)
for y in Ys:
    temp_data = Ys[y]
    print("%s set: sample size = %d, incidence rate = %f"
          % (y, len(temp_data), (temp_data == 'yes').sum() / len(temp_data)))
logger.info('Finished randomly splitting data set')

############################## Clean Dataset #################################################
logger.info('Start to clean dataset')
################# Numerical variable #################
logger.info("Start to clean numerical variables")

def getMissingProportion(X):
    colnames = X.columns
    missing_proportions = []
    for colname in colnames:
        missing_proportions.append(X[colname].isnull().sum()/float(X[colname].size))
    missing_proportions = np.array(missing_proportions)
    return missing_proportions


def drawMissingProportion(missing_proportions):
    plt.hist(missing_proportions, 20, normed=1, facecolor='green',alpha=0.75)
    plt.title("Missing value proportion histogram")
    plt.xlabel("Proportion of Missing Values")
    plt.ylabel("Frequency")
    plt.axis([0.0, 1.0, 0.0, 14.0])
    # plt.show()

# Count # of missing values of each feature
missing_proportions = getMissingProportion(trainX)
# drawMissingProportion(missing_proportions)

# a large portion of features are with all missing data (> 90%), we will move those features out from our dataset.
# we choose to remove features that have 80% of missing values
feature_to_remove = missing_proportions < 0.2
trainX = trainX.ix[:, feature_to_remove]

# only 76 features remained in the training set.
missing_proportions = getMissingProportion(trainX)
# drawMissingProportion(missing_proportions)

# Fill missing values of continuous variables with mean
float_features = [colname for colname in trainX.columns if trainX[colname].dtype == float]
trainX_mean = trainX.mean()
trainX_std = trainX.std()

# Replacing missing values with mean of train set, and standardizing numerical variables
for colname in float_features:
    missing_values = trainX[colname].isnull()
    if missing_values.sum() > 0:
        trainX.ix[missing_values.tolist(), colname] = trainX_mean[colname]
    trainX.ix[:, colname] = (trainX[colname] - trainX_mean[colname]) / trainX_std[colname]

    missing_values = validateX[colname].isnull()
    if missing_values.sum() > 0:
        validateX.ix[missing_values.tolist(), colname] = trainX_mean[colname]
    validateX.ix[:, colname] = (validateX[colname] - trainX_mean[colname]) / trainX_std[colname]

    missing_values = testX[colname].isnull()
    if missing_values.sum() > 0:
        testX.ix[missing_values.tolist(), colname] = trainX_mean[colname]
    testX.ix[:, colname] = (testX[colname] - trainX_mean[colname]) / trainX_std[colname]

# Double check whether missing values of every feature has been filled
print(trainX[float_features].isnull().sum())
logger.info('Finished cleaning numerical variables')

################# Categorical variable #################
logger.info('Start to clean categorical variables')
def replaceMissing(df, category_features):
    if category_features is None:
        category_features = [colname for colname in df.columns if df[colname].dtype != float]
    for feature in category_features:
        missing_values = df[feature].isnull()
        if missing_values.sum() > 0:
            df[feature].cat.add_categories('MISSING', inplace=True)
    return

def collapseCategories(df, category_features):
    if category_features is None:
        category_features = [colname for colname in df.columns if df[colname].dtype != float]
    collpased_features = {}
    for feature in category_features:
        col = df[feature].copy()
        for category in col.cat.categories:
            category_index = col == category
            if category_index.sum() < 0.05 * len(col):
                if feature not in collpased_features:
                    collpased_features[feature] = []
                collpased_features[feature].append(category)

                if 'OTHERS' not in df[feature].cat.categories:
                    df[feature].cat.add_categories('OTHERS', inplace=True)
                df.ix[category_index, feature] = 'OTHERS'
                df[feature].cat.remove_categories(category, inplace=True)
    return collpased_features

def getGoodCategories(df, category_features):
    if category_features is None:
        category_features = [colname for colname in df.columns if df[colname].dtype != float]
    feature_to_remove = set()
    for feature in category_features:
        categories = df[feature].cat.categories
        if len(categories) == 1 or len(set(categories) - set(['MISSING', 'OTHERS'])) < 2:
            feature_to_remove.add(feature)
    return list(set(category_features) - feature_to_remove)

def collapseGivenCategories(df, collapse_features):
    for feature in collapse_features:
        if 'OTHERS' not in df[feature].cat.categories:
            df[feature].cat.add_categories('OTHERS', inplace=True)

        col = df[feature].copy()
        for category in collapse_features:
            category_index = col == category
            if category_index.sum() != 0:
                df.ix[category_index, feature] = 'OTHERS'
                df[feature].cat.remove_categories(category, inplace=True)
    return

category_features = [colname for colname in trainX.columns if trainX[colname].dtype != float]

# Check how many categories each categorical variable has
category_features_levels = trainX[category_features].apply(lambda col: len(col.cat.categories))
print(category_features_levels)

# Some categorical variables are with too many categories, we can exclude those features out from the dataset
category_features = category_features_levels[category_features_levels <= 500].index
print(category_features)

replaceMissing(trainX, category_features)
collapse_features = collapseCategories(trainX, category_features)

# Same process to clean validation set and test set
replaceMissing(validateX, category_features)
collapseGivenCategories(validateX, collapse_features)

replaceMissing(testX, category_features)
collapseGivenCategories(testX, collapse_features)

# Use collpase_features of train set to collapse features of validation set and test set
category_features = getGoodCategories(trainX, category_features)
print(category_features)

logger.info('Finished to clean categorical variables')

############################## Feature Engineering #################################################
logger.info('Start to do feature engineering')
features = list(float_features) + list(category_features)
print('Total number of features remained: %d' % (len(features)))

trainX = trainX[features]
validateX = validateX[features]
testX = testX[features]

print(trainX.shape, validateX.shape, testX.shape)

X_all = pd.concat([trainX, validateX, testX])
print(X_all.shape)
X_dummy = pd.get_dummies(X_all)
print(X_dummy.shape)

trainX_dummy = X_dummy[:len(trainX)].as_matrix()
validateX_dummy = X_dummy[len(trainX):(len(trainX) + len(validateX))].as_matrix()
testX_dummy = X_dummy[(len(trainX) + len(validateX)):].as_matrix()

# PCA to combine features: Need to justify the reason to choose n_components = 32
pca = PCA(n_components=32)
trainX = pca.fit_transform(trainX_dummy)
print(trainX.shape)
logger.info('Total number of features after PCA: %d' % trainX.shape[1])
logger.info("Variance explained by %d principle components is: %f"
            % (trainX.shape[1], pca.explained_variance_ratio_.sum()))

trans_mat=np.transpose(pca.components_)
validateX = np.dot(validateX_dummy, trans_mat)
testX = np.dot(testX_dummy, trans_mat)
print(validateX.shape)
print(testX.shape)
logger.info('Finished feature engineering through PCA')

############################## Build Model #################################################
# Train and test your model using trainX, validateX, testX