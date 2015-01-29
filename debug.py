import cv2
import cPickle
import numpy as np
import pickle
from CNN import CNN

### check train set
# f=open('dataset/train_set')
# train_set=cPickle.load(f)
# f.close()
# pos=[]
# print sum(train_set[1])
# for i in range(train_set[1].shape[0]):
#     if train_set[1][i]:
#         pos.append(train_set[0][i])
#         # img=train_set[0][i].reshape((3,60,30))
#         # img=cv2.merge((img[0],img[1],img[2]))
#         # cv2.imshow('img',img)
#         # cv2.waitKey()
# pos=np.array(pos)
# print pos.shape
# f=open('dataset/pos_set','wb')
# cPickle.dump(pos,f)
# f.close()


### check overfitting
cnn=CNN(mode='predict',params_path='params',batch_size=10000)
f=open('dataset/train_set')
train_set=cPickle.load(f)
f.close()
print '---predict'
result=cnn.predict(train_set[0][:10000])
print np.sum(np.abs(result-train_set[1][:10000]))

# cnn=CNN(type='debug',params_path='params',batch_size=1)
# f=open('dataset/train_set')
# train_set=cPickle.load(f)
# f.close()
#
# for i in range(train_set[0].shape[0]):
#     cnn_input= cnn.cnn_input(train_set[0][i].reshape(1,5400)/255.)
#     cnn_input*=255
#     cnn_input=np.array(cnn_input,dtype=np.uint8)
#     print train_set[1][i]
#     cv2.imshow('img',cv2.merge((cnn_input[0][0],cnn_input[0][1],cnn_input[0][2])))
#     cv2.waitKey()

# f=open('backup/params_1')
# params=pickle.load(f)
# f.close()
#
# param_1=params[-2].get_value()
# for i in range(param_1.shape[0]):
#     for j in range(param_1.shape[1]):
#         weights=param_1[i][j]
#         min_value=np.min(weights)
#         weights-=min_value
#         print weights
#         cv2.imshow('weight',cv2.resize(weights,(50,50)))
#         cv2.waitKey()