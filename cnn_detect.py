import pyopencl as cl
from time import time
import cv2
import numpy as np
from CNN import CNN

class cnn_detect():
    def __init__(self,batch_size):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices(device_type=cl.device_type.GPU)[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags
        f = open('copy_vecs.cl', 'r')
        fstr = "".join(f.readlines())
        self.prg = cl.Program(self.ctx, fstr).build()

        self.cnn = CNN('predict', 'params', batch_size=batch_size)

        # print self.platform.name
        # print self.device.name


    def cnn_detect(self, frame):
        vec_list = None
        pos_list = None
        for scale, top, bottom in ((180, 0, 420), (240, 180, 720), (300, 420, 1080), (350, 720, 1080)):
            band = frame[top:bottom]
            scale = scale / 60.
            width = int(band.shape[1] / scale)
            height = int(band.shape[0] / scale)
            band = cv2.resize(band, (width, height))
            band = np.array(band, dtype=np.float32)
            # band_b,band_g,band_r=cv2.split(band)
            # band_tmp=np.array((band_b,band_g,band_r))
            rows = len(range(0, height - 60, 4))
            cols = len(range(0, width - 30, 3))
            local_vec_list = np.empty((rows * cols, 5400), dtype=np.float32)
            local_pos_list = np.empty((rows * cols, 4), dtype=np.float32)
            band_buf = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=band)
            vec_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, local_vec_list.nbytes)
            pos_buf = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, local_pos_list.nbytes)
            self.prg.copy_vecs(
                self.queue, (rows, cols), None,
                np.int32(band.shape[0]), np.int32(band.shape[1]),  # band rows, band cols
                np.int32(rows), np.int32(cols),
                np.float32(scale), np.float32(top),
                band_buf, vec_buf, pos_buf
            ).wait()
            cl.enqueue_read_buffer(self.queue, vec_buf, local_vec_list).wait()
            cl.enqueue_read_buffer(self.queue, pos_buf, local_pos_list).wait()
            # cv2.imshow('band',np.array(band,dtype=np.uint8))
            # for vec in local_vec_list:
            # vec=np.array(vec.reshape(3,60,30)*255.,dtype=np.uint8)
            # cv2.imshow('vec',cv2.merge((vec[0],vec[1],vec[2])))
            # cv2.waitKey()
            if vec_list == None:
                vec_list = local_vec_list
            else:
                vec_list = np.vstack((vec_list, local_vec_list))
            if pos_list == None:
                pos_list = local_pos_list
            else:
                pos_list = np.vstack((pos_list, local_pos_list))
        result = self.cnn.predict(vec_list)
        return result, pos_list
