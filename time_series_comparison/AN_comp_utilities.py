# includes functions used to assess similarity of time series provided

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
