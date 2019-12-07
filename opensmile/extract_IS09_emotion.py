import os
import time
from datasets.IemocapDataset import IemocapDataset

start_time = time.time()

# Specify output file name and the file format that SMILEXtract will output based on the given opensmile configuration file
output_file_name = 'output_2'
output_file_ext = '.arff'

# Load dataset
dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')

# Remove output file if exists
if os.path.exists(output_file_name + output_file_ext):
    os.system('rm ' + output_file_name + output_file_ext)

# Iterate through dataset and extract features using SMILExtract for each sample
for i in range(len(dataset)):
    execute_string = 'SMILExtract -C config/IS09_emotion.conf -I ' + dataset[i]['path'] + ' -O ' + output_file_name + output_file_ext
    os.system(execute_string)

end_time = time.time()
total_time = end_time - start_time
print('Program took %d min %d sec to complete' % (total_time // 60, total_time % 60))