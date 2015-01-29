import cv2
import numpy as np

class Human:
    def __init__(self,box,gray_frame):
        self.box=box
        self.template=gray_frame[self.box[1]:self.box[3],self.box[0]:self.box[2]]
        self.age=0
        self.last_shift_x=0.
        self.last_shift_y=0.

    def update(self,gray_frame):
        self.age+=1
        if self.age>80:
            return False
        try:
            center_x=(self.box[0]+self.box[2])/2.
            center_y=(self.box[1]+self.box[3])/2.
            width=self.box[2]-self.box[0]
            height=self.box[3]-self.box[1]

            search_img=gray_frame[center_y-0.75*height:center_y+0.75*height,center_x-0.75*width:center_x+0.75*width]

            res = cv2.matchTemplate(search_img,self.template,cv2.TM_SQDIFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            new_x=center_x-0.75*width+min_loc[0]+0.4*self.last_shift_x
            new_y=center_y-0.75*height+min_loc[1]+0.4*self.last_shift_y
            self.last_shift_x=new_x-self.box[0]
            self.last_shift_y=new_y-self.box[1]
            if np.abs(self.last_shift_x)>width/9.:
                return False
            if np.abs(self.last_shift_y)>height/9.:
                return False
            self.box[0]=new_x
            self.box[1]=new_y
            self.box[2]=self.box[0]+width
            self.box[3]=self.box[1]+height
            if self.age%5==0:
                self.template=gray_frame[self.box[1]:self.box[3],self.box[0]:self.box[2]]
            return True
        except:
            return False

    def get_box(self):
        return self.box

    def get_points(self):
        return self.points

    def is_included(self,box):
        center_x=(self.box[0]+self.box[2])/2.
        center_y=(self.box[1]+self.box[3])/2.
        width=1.1*(self.box[2]-self.box[0])
        height=1.1*(self.box[3]-self.box[1])
        if box[0]>(center_x-width/2.) and box[1]>(center_y-height/2.) \
            and box[2]<(center_x+width/2.) and box[3]<(center_y+height/2.):
            return True
        box_c_x=(box[0]+box[2])/2.
        box_c_y=(box[1]+box[3])/2.
        if np.abs(box_c_x-center_x)<width/3. and np.abs(box_c_y-center_y)<height/3.:
            return True
        return False

    def draw(self,frame):
        cv2.rectangle(frame,(self.box[0],self.box[1]),(self.box[2],self.box[3]),(255,100,100),2)