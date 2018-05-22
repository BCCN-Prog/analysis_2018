# Includes functions used to assess similarity of time series provided

# Weather can be though of as a non-stationary object in high dimensional space.
# Specific trajectories on this space correspond to specific realisations of weather
# conditions (temperature, wind, humidity and plenty more) across time.
# The perfect weather prediction machine recreates this high dimensional space, and
# can thus predict the unfolding of the different phenomena. However, weather is a
# chaotic system, meaning that small differences in initial conditions can lead to huge
# differences in the way the phenomena evolve as time passes. ALso, it is a non-
# stationary system, meaning that the statistics affecting the form of the high-dimensional
# object change with time.

# Since it is so difficult to reconstruct this whole space, predictive models try
# to reduce it. Also, the ultimate goal is to have a comprehensive picture of
# specific variables, like the temperature, which can be reported and understood by people.
# So, the assessment of whether a particular model is good or bad makes sense only in
# the low dimensional space of general interest. Ultimately, we want to receive a one dimensional
# time series created by a certain predictive model, compare it to the real data,
# and tell whether the prediction was good enough.

# Defining what is a good prediction can be quite complex and depends on human factors
# (for example, how much a 10 % difference in humidity affects felt temperature?) that
# are not available to us. The goal here is to provide a framework of different techniques
# and measures, and also characterize each one in terms of what errors it punishes more
# or less severely.

# Interesting article: http://dbgroup.eecs.umich.edu/files/sigmod07timeseries.pdf
# The aforementioned point also stated in the article: " Time
# series simularity methods can be used for computing trajectory similarity"
# http://www.cs.ust.hk/~leichen/pub/04/vldb04.pdf
# if triangular identity doesn't hold then distance stops being a local feature, so
# is doesn't make sense to compare distances (maybe there are better paths)

# DTW measures a distance-like quantity between two given sequences, but it doesn't guarantee the triangle inequality to hold.

# https://infolab.usc.edu/csci599/Fall2003/Time%20Series/Efficient%20Similarity%20Search%20In%20Sequence%20Databases.pdf
# "The euclidean distance measure is the optimal distance measure for estimation if the signals are corrupted by Gaussian, additive noise."

# To introduce a measure that is independent of the length of the data we are trying
# to predict, a way to find how the maximum of a gaussian noise array is changing
# relating to the size of the array. For sure the dependence on n is quite weak,
# especially as n grows a lot (sqrt(n) was tried but it is too fast). Not sure if
# such a prediction exists at all

# Grigoriy Vevyurko found the solution:
# https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables
# find the probability distibution of all measurements being below a value, then the complementary, then get the pdf and finaly compute the mean
# on this and use it to scale the measure

import numpy as np
from scipy.stats import norm

def compare_time_series(prediction, true, days_ahead, method, data_type = 'temperature'):

    # Compares predicted and true time series and produces a measure of similarity.
    # This measure can judge how good the prediction was.

    # In:
    # prediction : array of length n with predicted values for a specific number of days ahead
    # true : array of same length n with true values
    # days_ahead : how many days ahead was the prediction made? Could weight final measure with this
    # method : measure of similarity used
    # data_type : data type of arrays. Changes the way the measure of similarity is scaled.
    # Could be 'temperature' in degrees Celcius, 'humidity' as a percentage,
    # 'precipitation' in mm, 'wind' in km/h

    # Out:
    # measure : a measure of similarity as a single number between 0 and 1. scaling of the measure
    # is method-dependent, so it will be done inside the function for each method
    # value : the exact value of the measure used, if such exists (unbiased measure)

    # It is important to tell apart the predicted time series from the true one,
    # because not all measures of similarity are symmetric (non-metric operators)

    if len(prediction) != len(true):
        print('Arrays to be compared do not have the same length!')
        return 0

    measure, value = method(prediction, true, data_type)

    # here a way to scale measure based on days_ahead (independently of data_type?)
    # bends the measure which is between 0 and 1 so that prediction of many days ahead
    # with the same performance based on 'method' are viewed in a better light
    # maybe forth root is a little bit harsh

    measure = measure ** np.sqrt(np.sqrt(days_ahead))

    return measure, value



def variance(prediction, true, data_type):
    # find variance as a global descriptive measure of similarity

    var = np.sqrt( ((prediction - true) ** 2).mean() )
    # scale variance by mean
    measure = var/true.mean()
    # scale result according to what data type we have
    if data_type == 'temperature':
        b = 2
    elif data_type == 'humidity':
        b = 3
    elif data_type == 'wind':
        b = 2
    elif data_type == 'precipitation':
        b = .6
    # convert result to a value between 0 and 1, using tanh
    measure = np.tanh(b*measure)

    return measure, var


def norm1(prediction, true, data_type):
    # find first norm as a global descriptive measure of similarity
    # Punishes outliers less severely, not the optimal choice if the difference
    # signal is gaussian noise (which it should be)

    norm = abs(prediction - true).mean()
    # scale variance by mean
    measure = norm/true.mean()
    # scale result according to what data type we have
    if data_type == 'temperature':
        b = 2.5
    elif data_type == 'humidity':
        b = 4
    elif data_type == 'wind':
        b = 2.5
    elif data_type == 'precipitation':
        b = .8
    # convert result to a value between 0 and 1, using tanh
    measure = np.tanh(b*measure)

    return measure, norm


def outlier(prediction, true, data_type):
    # find the biggest outlier as a local descriptive measure of similarity
    # With weather, we are interested in a model that is not necessarily very exact,
    # but when it fails it does not fail hard. For example, we do not want a prediction
    # for a single day to be zero precipitation, and it turns out raining, even if this model
    # is near perfect for the other days

    outlier = max(abs(prediction-true))
    # unbiased variance estimator
    n = len(prediction)
    var_est = np.var(prediction-true,ddof = 1)
    # expected outlier if we assume that the difference is gaussian noise
    exp_outlier = var_est*np.sqrt(np.log10(n**2/(2*np.pi*np.log10(n**2/(2*np.pi)))))*(1+0.577/np.log10(n))
    # scale outlier by expected outlier for this size of arrays
    outlier_sc = outlier*(4*var_est/exp_outlier)
    # scale outlier by mean
    measure = outlier_sc/true.mean()
    # scale result according to what data type we have
    if data_type == 'temperature':
        b = .5
    elif data_type == 'humidity':
        b = 1
    elif data_type == 'wind':
        b = .5
    elif data_type == 'precipitation':
        b = .3
    # convert result to a value between 0 and 1, using tanh
    measure = np.tanh(b*measure)

    return measure, outlier

def convert_rainfall(prob_rain,mean,variance):
    # prob_rain is the probability of rain as predicted by the weather forecast
    # mean is the mean rainfall in mm (for the month?), taken fron the data or computed on the data
    # for this specific station
    # variance is the variance of the rainfall in mm for the month

    # fit with a gaussian distribution
    rainfall = mean + variance*norm.ppf(prob_rain)
    # limit values to logical ones
    rainfall = max(min(rainfall, mean + 3*variance), 0)
    if prob_rain<.1:
        rainfall = 0
    return rainfall
