
import os
import fnmatch
import re 
import scipy.io

hardsamples = fnmatch.filter(os.listdir('../data_train/train/'),'im20000*mat')
# labels = os.path.join(path_train,'train')
hs_IDs=[]
for hs in hardsamples:
    hs_IDs.append(int(re.findall('[0-9]+',hs)[0]))
# print (hs_IDs)

gtimg_name = os.path.join('../data_train/','clean','I' + str(hs_IDs[0]) + '.mat')
gtimg_mat = scipy.io.loadmat(gtimg_name)
