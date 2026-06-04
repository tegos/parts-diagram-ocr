import numpy as np
import pandas as pd
from skimage.filters import gaussian_filter
from skimage.filters import sobel
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# function to carry out feature engineering methods on input data and return these features
def feature_engineering(data, flag, pca_fit_data=None, pca_test_flag=False, n_pca=175, std_blur=1.0,
                        pixels_per_cell_hog=(4, 4), orientations_hog=5):
    X = data.copy()
    # compute the number of pixel rows/columns in the image
    image_size = int(np.sqrt(X.shape[1]))

    # method to replace each image in the dataset with the gaussian blurred version of itself
    if (flag == 'gaussian blur'):
        for i in X.index:
            # retrive the ith observation
            obs_unfiltered = np.array(X.iloc[i], dtype=np.float64)
            # resize the current observation into a matrix
            obs_unfiltered = np.resize(obs_unfiltered, new_shape=(image_size, image_size))
            # compute the filtered image
            obs_filtered = gaussian_filter(obs_unfiltered, std_blur)
            # reshape the filtered image into an array
            obs_filtered = np.resize(obs_filtered, new_shape=(image_size * image_size))
            # store the filtered image back into the data frame
            X.iloc[i] = obs_filtered

    elif (flag == 'sobel edge detection'):
        for i in X.index:
            # retrive the ith observation
            obs_unfiltered = np.array(X.iloc[i], dtype=np.float64)
            # resize the current observation into a matrix
            obs_unfiltered = np.resize(obs_unfiltered, new_shape=(image_size, image_size))
            # compute the filtered image
            obs_filtered = sobel(obs_unfiltered)
            # reshape the filtered image into an array
            obs_filtered = np.resize(obs_filtered, new_shape=(image_size * image_size))
            # store the filtered image back into the data frame
            X.iloc[i] = obs_filtered
    elif (flag == 'principal components'):
        std_scaler = StandardScaler()
        # if applying pca to training data
        if (pca_test_flag == False):
            X = pd.DataFrame(std_scaler.fit_transform(X))
            # initialise a pca model
            pca = PCA(n_components=n_pca)
            # fit the model and compute PCs
            X = pd.DataFrame(pca.fit_transform(X))
        # if applying pca to test data, need to fit model on training data
        else:
            std_scaler.fit(pca_fit_data)
            X = pd.DataFrame(std_scaler.transform(X))
            pca_fit_data = pd.DataFrame(std_scaler.transform(pca_fit_data))
            # initialise a pca model
            pca = PCA(n_components=n_pca)
            # fit the model on training data and compute PCs on test data
            pca.fit(pca_fit_data)
            X = pd.DataFrame(pca.transform(X))

    elif (flag == 'hog features'):
        for i in X.index:
            # retrive the ith observation
            obs_unfiltered = np.array(X.iloc[i], dtype=np.float64)
            # resize the current observation into a matrix
            obs_unfiltered = np.resize(obs_unfiltered, new_shape=(image_size, image_size))
            # compute the filtered image
            obs_filtered_features, obs_filtered_image = hog(obs_unfiltered, orientations=orientations_hog,
                                                            pixels_per_cell=pixels_per_cell_hog, visualise=True)
            # reshape the filtered image into an array
            obs_filtered = np.resize(obs_filtered_image, new_shape=(image_size * image_size))
            # store the filtered image back into the data frame
            X.iloc[i] = obs_filtered

    else:
        return 'Error: feature engineering method not found'

    return X


############################################################################
############################################################################

# Read in MNIST digits data
Digits_train_data = pd.read_csv("C:\Chris\Data Science\Kaggle\DigitRecogniser\train.csv", header=0)
Digits_test_data = pd.read_csv("C:\Chris\Data Science\Kaggle\DigitRecogniser\test.csv", header=0)
# take a copy of the training data

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(Digits_train_data.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(Digits_test_data.shape))
# Any files you write to the current directory get shown as outputs

X_all = Digits_train_data.copy()
# take a sample of the training data to use for CV
X_sample = X_all.sample(n=20000, replace=True, random_state=1)
X = X_sample.copy()
# reset index for the sample
X.index = range(len(X))
# separate labels from data in training set
t = X['label']
del X['label']
# Transform data using one of the feature engineering methods
# X = feature_engineering(X, flag = 'gaussian blur')
# X = feature_engineering(X, flag = 'sobel edge detection')
# X = feature_engineering(X, flag = 'hog features')

# initialise folds for 10 fold CV
folds = KFold(len(X), n_folds=10)
# create array to store accuracy scores
accuracy_scores = [0] * 10
# initialise counter for iterating in the scores array
i = 0
# fit and score model for each CV fold
for train_index, test_index in folds:
    # for the currect fold, retrieve training data and preprocess it
    X_train_unprocessed = X.iloc[train_index]
    X_train = feature_engineering(X_train_unprocessed, flag='principal components')
    # X_train = X_train_unprocessed
    # retrieve training target
    t_train = t.iloc[train_index]
    # retrive testing data and preprocess it
    X_test_unprocessed = X.iloc[test_index]
    X_test = feature_engineering(X.iloc[test_index], flag='principal components', pca_fit_data=X_train_unprocessed,
                                 pca_test_flag=True)
    # X_test = X_test_unprocessed
    # retrive test target
    t_test = t.iloc[test_index]

    # initialise model
    SVM = LinearSVC(C=0.2)
    # fit model
    SVM.fit(X_train, t_train)
    # predict on test data
    t_predict = SVM.predict(X_test)
    # compute accuracy of current fold and store in array
    accuracy_scores[i] = accuracy_score(t_test, t_predict)
    # increment counter
    i += 1

    # print i, '\n'

# compute mean and std of accuracy score
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

# take a copy of the testing data
X_testing_data = Digits_test_data.copy()
# reset the training data as the whole dataset
X = X_all
# separate labels from data in training set
t = X['label']
del X['label']
# preprocess submission data
# X_processed = X
# X_testing_data_processed = X_testing_data

# X_processed = feature_engineering(X, flag = 'gaussian blur')
# X_testing_data_processed = feature_engineering(X_testing_data, flag = 'gaussian blur')

# X_processed = feature_engineering(X, flag = 'sobel edge detection')
# X_testing_data_processed = feature_engineering(X_testing_data, flag = 'sobel edge detection')

# X_processed = feature_engineering(X, flag = 'hog features')
# X_testing_data_processed = feature_engineering(X_testing_data, flag = 'hog features')

X_processed = feature_engineering(X, flag='principal components')
X_testing_data_processed = feature_engineering(X_testing_data, flag='principal components', pca_fit_data=X,
                                               pca_test_flag=True)

# initialise a model for submission
SVM_sub = LinearSVC()
# fit model on training dataset
SVM_sub.fit(X_processed, t)
# compute predictions on submission data
t_predict_sub = pd.DataFrame(SVM_sub.predict(X_testing_data_processed))
# format and write the predictions to a csv file
t_predict_sub.index += 1
t_predict_sub.index.names = ['ImageId']
t_predict_sub.columns = ['Label']
t_predict_sub.to_csv(path_or_buf='results.csv')
