import cv2
import numpy as np
from time import time
from sklearn.cluster import MeanShift
from Human import Human
from multiprocessing import Process, Queue,Lock
from copy import deepcopy
from cnn_detect import cnn_detect
from random import randint

def detect_and_track():
    detector = cnn_detect(9154)
    frame_queue=Queue()
    humans_queue=Queue()
    gray_frame_queue=Queue()
    result_queue=Queue()
    Process(target=track_process,
            args=(frame_queue,humans_queue,gray_frame_queue,result_queue,)
    ).start()
    while True:
        frame = frame_queue.get()
        humans_copy = humans_queue.get()
        start_time=time()
        ret, pos_list = detector.cnn_detect(frame)
        pos_list = pick_pos(ret, pos_list)
        pos_list = reduce_duplicate(pos_list, frame)
        local_human = add_human(pos_list, frame, humans_copy)
        while not gray_frame_queue.empty():
            a_frame = gray_frame_queue.get()
            tmp_humans = []
            for human in local_human:
                ret = human.update(a_frame)
                if ret == True:
                    tmp_humans.append(human)
            local_human = tmp_humans
        result_queue.put(local_human)
        print '\tdetect time :',time()-start_time


def track_process(frame_queue,humans_queue,gray_frame_queue,result_queue):
    # first frame, detect human and store into list
    frame_num = 0
    cap = cv2.VideoCapture('a.avi')

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1920,1080))

    ret, frame = cap.read()
    frame_queue.put(frame)
    humans = []
    humans_queue.put(deepcopy(humans))
    humans = result_queue.get()
    # flag_detect : True -> is running detect thread ; False -> not
    flag_detect = False
    while (cap.isOpened()):
        start_time=time()
        frame_num += 1
        print 'frame_num :', frame_num
        ret, frame = cap.read()
        humans = track(frame, humans)

        if flag_detect == False:
            humans_copy = deepcopy(humans)
            frame_queue.put(frame)
            humans_queue.put(humans_copy)
            flag_detect = True
        else:
            if result_queue.empty():
                gray_frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                humans = result_queue.get()
                humans = track(frame, humans)
                flag_detect = False

        tmp_frame=draw(frame, humans)
        cv2.imshow('tmp3_frame', cv2.resize(tmp_frame, (800, 450)))
        out.write(tmp_frame)
        during_time=time()-start_time
        print 'during_time :',during_time
        wait_time=50-1000*during_time
        if wait_time>0:
            cv2.waitKey(int(wait_time)+1)

    cap.release()
    out.release()


def track(frame, humans):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    local_humans = []
    for human in humans:
        ret = human.update(gray_frame)
        if ret == True:
            local_humans.append(human)
    return local_humans


def draw(frame, humans):
    tmp_frame = np.copy(frame)
    for human in humans:
        human.draw(tmp_frame)
    return tmp_frame


def pick_pos(result, pos_list):
    new_pos_list = None
    for i in range(result.shape[0]):
        if result[i] == 1:
            if new_pos_list == None:
                new_pos_list = np.array([pos_list[i]])
            else:
                new_pos_list = np.vstack((new_pos_list, pos_list[i]))
    return new_pos_list


def reduce_duplicate(pos_list, frame):
    cluster_centers = None
    if pos_list != None:
        tmp_pos_list = list(pos_list)
        s_pos = [pos for pos in tmp_pos_list if pos[3] - pos[1] < 225]
        l_pos = [pos for pos in tmp_pos_list if pos[3] - pos[1] > 225]
        if len(s_pos) > 0:
            s_pos = np.array(s_pos)
            bandwidth = 40
            ms = MeanShift(bandwidth)
            ms.fit(s_pos)
            cluster_centers = ms.cluster_centers_
        if len(l_pos) > 0:
            l_pos = np.array(l_pos)
            bandwidth = 60
            ms = MeanShift(bandwidth)
            ms.fit(l_pos)
            cluster_centers = np.vstack((cluster_centers, ms.cluster_centers_))
        cluster_centers = sorted(cluster_centers, key=lambda vec: vec[3], reverse=True)

        # tmp_frame=np.copy(frame)
        # for pos in cluster_centers:
        #     cv2.rectangle(tmp_frame,
        #                   (int(pos[0]),int(pos[1])),
        #                   (int(pos[2]),int(pos[3])),
        #         (randint(0,255),randint(0,255),randint(0,255)),2)
        # cv2.imshow('tmp_frame',cv2.resize(tmp_frame,(800, 450)))
        # cv2.waitKey()

    return cluster_centers


def add_human(pos_list, frame, humans):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    local_humans = []
    if pos_list:
        for box in pos_list:
            is_included = False
            for human in local_humans:
                if human.is_included(box):
                    is_included = True
                    break
            if is_included == False:
                local_humans.append(Human(box, gray_frame))
    for human in humans:
        is_included = False
        for human_2 in local_humans:
            if human_2.is_included(human.get_box()):
                is_included = True
                break
        if is_included == False:
            local_humans.append(human)
    return local_humans


if __name__ == '__main__':
    detect_and_track()