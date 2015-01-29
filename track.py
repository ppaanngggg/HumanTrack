import cv2
import numpy as np
from Human import Human

video=cv2.VideoCapture('TownCentreXVID.avi')

lk_params=dict(
    winSize=(15,15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_COUNT,10,0.03)
)

ret,frame=video.read()
prev_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
mask=np.zeros_like(prev_frame)
mask[378:628,854:982]=255
points=cv2.goodFeaturesToTrack(prev_frame,mask=mask,maxCorners=10,qualityLevel=0.01,minDistance=7)
human=Human([[854,378],[982,628]],points)
box=human.get_box()
cv2.rectangle(frame,(box[0][0],box[0][1]),(box[1][0],box[1][1]),(0,255,255),2)
for point in human.get_points():
    cv2.circle(frame,(point[0][0],point[0][1]),3,(0,0,255),-1)
cv2.imshow('frame',frame)
# cv2.waitKey()
while video.isOpened():
    ret,frame=video.read()
    next_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    human.update(prev_frame,next_frame,lk_params)
    box=human.get_box()
    cv2.rectangle(frame,(int(box[0][0]),int(box[0][1])),(int(box[1][0]),int(box[1][1])),(0,255,255),2)
    for point in human.get_points():
        cv2.circle(frame,(point[0][0],point[0][1]),3,(0,0,255),-1)
    cv2.imshow('frame',cv2.resize(frame,(800,450)))
    cv2.waitKey()
    # if cv2.waitKey(10) & 0xff==ord('q'):
    #     break
    prev_frame=next_frame
video.release()