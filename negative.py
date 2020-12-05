
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import threading


import _init_paths
import os
import cv2


			
def run_video():

	cam = cv2.VideoCapture(0)
	# cam.set(cv2.CAP_PROP_AUTOFOCUS, opt.cam_focus)
	# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.cam_res[1])
	# cam.set(cv2.CAP_PROP_FRAME_WIDTH, opt.cam_res[0])
	i=0
	while True:
		_, img = cam.read()
		filename="sample_{}".format(i)
		cv2.imshow(filename,img)
		cv2.imwrite(os.path.join('E:/KAERI_DATAnegative_sample/',filename+".png", v)
		i+=1
		if cv2.waitKey(1) == 27:
			break

	cam.release()
	print("close")

	

	

if __name__ == '__main__':
	run_video()	
