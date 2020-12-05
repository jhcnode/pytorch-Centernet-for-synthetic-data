

import sys
import _init_paths
import os
import cv2





	
			
def run_video():


	cam = cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
	save_filename="test.avi"
	save_dir='./test_data/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	vid_h=int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))			
	vid_w=int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))	
	fps= int(cam.get(cv2.CAP_PROP_FPS))	
	vid_writer= cv2.VideoWriter(os.path.join(save_dir,save_filename), fourcc, fps, (vid_w,vid_h))	


	while True:
		_, img = cam.read()
	
		if(img is not None):
			cv2.imshow('cam',img)
			vid_writer.write(img)

		if cv2.waitKey(1) == 27:
			break

	vid_writer.release()
	cam.release()
	print("close")



if __name__ == '__main__':
	run_video()	
