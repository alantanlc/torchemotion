import os
import time
from datasets.IemocapDataset import IemocapDataset

# Log start time
start_time = time.time()

# Specify file path and name (include file extension)
config_file_path = 'config/IS09_emotion.conf'
output_file_name = 'iemocap_is09_emotion.csv'

# Load dataset
dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')

# Remove output file if exists
if os.path.exists(output_file_name):
    os.system('rm ' + output_file_name)

# Iterate through dataset and extract features of each sample using SMILExtract
# For os.system(execute_string) to work on Linux:
# 1. Ensure that SMILExtract executable exists in /usr/local/bin.
# 2. Ensure that SMILExtract executable can be executed without sudo.
#    - This can be achieved by granting all users with the permission to execute the SMILExtract executable using the following command: chmod a+x /usr/local/bin/SMILExtract
for i in range(len(dataset)):
    execute_string = 'SMILExtract -C ' + config_file_path + ' -I ' + dataset[i]['path'] + ' -csvoutput ' + output_file_name
    os.system(execute_string)

# Compute and print program execution time
end_time = time.time()
total_time = end_time - start_time
print('Program took %d min %d sec to complete' % (total_time // 60, total_time % 60))