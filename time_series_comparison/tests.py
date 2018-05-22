import AN_comp_utilities as utils
import numpy as np
# import traj_dist.distance as tdist

# create random weathor conditions to scale parameters in methods according to
# what is subjectively assumed a "good prediction"

a = np.random.normal(1.5,2,1000)
a[a<0] = 0
b = a + np.random.normal(0,.5,1000)
b[b<0] = 0

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


measure, value = utils.compare_time_series(a,b,1,utils.variance,'precipitation')
print('Variance')
print(measure)
print(value)

measure, value = utils.compare_time_series(a,b,1,utils.norm1,'precipitation')
print('Norm1')
print(measure)
print(value)

measure, value = utils.compare_time_series(a,b,1,utils.outlier,'precipitation')
print('Outlier')
print(measure)
print(value)

measure, value = utils.compare_time_series(prob_rain,rain,1,utils.cross_entropy,'prob_rain')
print('Outlier')
print(measure)
print(value)

utils.histogram_probability_of_rain(prob_rain_sham,rain)
