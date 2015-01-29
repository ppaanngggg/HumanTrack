from CNN import CNN
import cPickle
import os
from random import randint
import cv2
import numpy as np

# print '--------------------------------'
# print '---- pick negs from library ----'
# print '--------------------------------'
#
# NEG_ADD=2500
#
# f = open('dataset/train_set', 'rb')
# train_set = cPickle.load(f)
# f.close()
# # f = open('dataset/valid_set', 'rb')
# # valid_set = cPickle.load(f)
# # f.close()
#
# fold = 'dataset/neg/'
# filenames = os.listdir(fold)
#
# cnn = CNN(
#     mode='predict', params_path='params', batch_size=1000
# )
# count_add = 0
# failed_vecs = []
# while count_add < NEG_ADD:
#     index = randint(0, len(filenames) - 1)
#     filename = filenames[index]
#     if filename[-4:] == '.png' or filename == '.jpg':
#         print ' ', filename
#         img = cv2.imread(fold + filename)
#         img=cv2.resize(img,(img.shape[1]/6,img.shape[0]/6))
#         print img.shape
#         # cv2.imshow('img',img)
#         # cv2.waitKey()
#         vecs = []
#         for i in range(1000):
#             row = randint(0, img.shape[0] - 60)
#             col = randint(0, img.shape[1] - 30)
#             b, g, r = cv2.split(img[row:row + 60, col:col + 30])
#             b = list(b.reshape((30 * 60)))
#             g = list(g.reshape((30 * 60)))
#             r = list(r.reshape((30 * 60)))
#             vecs.append(np.array(b + g + r))
#         vecs = np.array(vecs)
#         results = cnn.predict(vecs/255.)
#         # print results
#         # print results.shape
#         for i in range(results.shape[0]):
#             if results[i]==1:
#                 if count_add < NEG_ADD:
#                     count_add += 1
#                     failed_vecs.append(vecs[i])
#         print count_add
# failed_vecs=np.array(failed_vecs)
# print '    count_add :', count_add
# train_x = list(train_set[0])
# train_y = list(train_set[1])
# for vec in failed_vecs:
#     train_x.append(vec)
#     train_y.append(0)
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# print '    train_x.shape :', train_x.shape
# train_set = (train_x, train_y)
# f = open('dataset/train_set', 'wb')
# cPickle.dump(train_set, f)
# f.close()
#
print '--------------------------------'
print '---- pick poss from library ----'
print '--------------------------------'

POS_ADD=1000

f = open('dataset/train2_set', 'rb')
pos_set = cPickle.load(f)
f.close()
x=pos_set[0]/255.
cnn=CNN(
        mode='predict', params_path='params',
        batch_size=x.shape[0]
    )

new_pos=[]
result=list(cnn.predict(x))

print len(result)-sum(result),'/',len(result)

# tmp_pos_set=list(pos_set[0])
# count=0
# while count<POS_ADD:
#     index=randint(0,len(result)-1)
#     if result.pop(index)==0:
#         new_pos.append(tmp_pos_set.pop(index))
#         count+=1
# print count
#
# f = open('dataset/train_set', 'rb')
# train_set=cPickle.load(f)
# f.close()
# train_set_vec=np.array(new_pos+list(train_set[0]))
# train_set_target=np.array([1]*POS_ADD+list(train_set[1]))
# print train_set_vec.shape
# print train_set_target.shape
# f = open('dataset/train_set', 'wb')
# cPickle.dump((train_set_vec,train_set_target),f)
# f.close()

# print '--------------------------------'
# print '---- reorder ----'
# print '--------------------------------'
#
# f = open('dataset/train_set', 'rb')
# train_set = cPickle.load(f)
# f.close()
#
# old_vec=list(train_set[0])
# old_target=list(train_set[1])
# new_vec=[]
# new_target=[]
#
# while len(old_vec)>0:
#     index=randint(0,len(old_vec)-1)
#     new_vec.append(old_vec.pop(index))
#     new_target.append(old_target.pop(index))
#
# new_vec=np.array(new_vec)
# new_target=np.array(new_target)
# print new_vec.shape
# print new_target.shape
#
# f = open('dataset/train_set', 'wb')
# cPickle.dump((new_vec,new_target),f)
# f.close()