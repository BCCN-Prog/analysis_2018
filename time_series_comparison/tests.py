import AN_comp_utilities as utils
import numpy as np
# import traj_dist.distance as tdist

# create random weathor conditions to scale parameters in methods according to
# what is subjectively assumed a "good prediction"

a = np.random.normal(1.5,2,1000)
a[a<0] = 0
b = a + np.random.normal(0,.5,1000)
b[b<0] = 0

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
