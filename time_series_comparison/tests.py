import AN_comp_utilities as utils
import numpy as np
# import traj_dist.distance as tdist

# create random weathor conditions to scale parameters in methods according to
# what is subjectively assumed a "good prediction"

true = np.random.normal(1.5,2,1000)
true[true<0] = 0
prediction = true + np.random.normal(0,.5,1000)
prediction[prediction<0] = 0

# not realistic prediction data
p = 0.5
rain = np.random.choice(a=[False, True], size=1000, p=[p, 1-p])
prob_rain_sham = p*100 +  np.random.normal(0,2,1000)
prob_rain_sham[prob_rain_sham<0] = 0
prob_rain_sham[prob_rain_sham>100] = 100

# more realistic
prob_rain = rain*100 + np.random.normal(0,15,1000)
prob_rain[prob_rain<0] = 0
prob_rain[prob_rain>100] = 100

# load and use real data

real = 0
pred = 0
data_type = 'temperature'
days_ahead = 1

measure, value, differences, per = utils.compare_time_series(prediction,true,days_ahead,utils.variance,data_type)
print('Variance')
print(measure)
print(value)

measure, value, differences, per = utils.compare_time_series(prediction,true,days_ahead,utils.norm1,data_type)
print('Norm1')
print(measure)
print(value)

measure, value, differences, per = utils.compare_time_series(prediction,true,days_ahead,utils.outlier,data_type)
print('Outlier')
print(measure)
print(value)

measure, value, differences, per = utils.compare_time_series(prob_rain,rain,1,utils.cross_entropy,'prob_rain')
print('Cross-entropy')
print(measure)
print(value)

utils.plot_histograms_rain(prob_rain,rain)
