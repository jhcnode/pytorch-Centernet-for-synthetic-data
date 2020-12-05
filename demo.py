
import sys
import _init_paths
import os
import cv2
import torch
import numpy as np
import time

from opts import opts
from detectors.detector_factory import detector_factory
from models.model import create_model, load_model
from datasets.dataset_factory import get_dataset,read_data
from utils.debugger import Debugger
from utils.utils import AverageMeter
from pynput.keyboard import Listener,Key
import threading

key_store=set()
zoom_factor=0.1
vis_thresh=0.3
isRun=True

def keyPressEvent(key):
	global zoom_factor
	global vis_thresh
	
	key_store.add(key)
	interval1=0.1
	check1=False
	if Key.up in key_store and len(key_store)==1:
		zm=zoom_factor
		pre=zm+interval1
		if(pre>1):
			zoom_factor=1
		else:
			zoom_factor+=interval1		
		check1=True
	if	Key.down in key_store and len(key_store)==1:
		zm=zoom_factor
		pre=zm-interval1
		if(pre<0):
			zoom_factor=0
		else:
			zoom_factor-=interval1
		check1=True		
		
	if(check1==True):
		factor=zoom_factor
		print("[zoom_factor:{:.1f}]".format(factor))
		
	interval2=0.1
	check2=False
	if  all(i in key_store for i in [Key.ctrl_l,Key.up]):
		vt=vis_thresh
		pre=vt+interval2
		if(pre>0.9):
			vis_thresh=0.9
		else:
			vis_thresh+=interval2
		check2=True
			
	if  all(i in key_store for i in [Key.ctrl_l,Key.down]):
		vt=vis_thresh
		pre=vt-interval2
		if(pre<0):
			vis_thresh=0
		else:
			vis_thresh-=interval2	
		check2=True
		
	if(check2==True):
		factor=vis_thresh
		print("[vis_thresh:{:.1f}]".format(factor))	
		
	
def handleRelease(key):
	global isRun
	if key in key_store:
		key_store.remove(key)
   
	if key == Key.esc:
		isRun=False
		return isRun	


def zoom_io(img,zoom_factor):
	h,w,c=img.shape
	
	cy=int(h/2)
	cx=int(w/2)
	
	
	def lerp(p1,p2,d):
		return (1-d)*p1 + d*p2
		
	lower_bound=0.1
	w_lower_bound=lower_bound*w
	h_lower_bound=lower_bound*h
	
	w=int(lerp(w_lower_bound,w,1-zoom_factor))
	h=int(lerp(h_lower_bound,h,1-zoom_factor))
	cx=cx-int(w/2)
	cy=cy-int(h/2)
	img=img[cy:cy+h,cx:cx+w]
	return img
def run_input():

	listener=Listener(on_press=keyPressEvent,on_release=handleRelease)
	listener.start()
	listener.join()



def read_map(data_dir):
	label_map_dir = os.path.join(data_dir, "map.txt")
	label_map=[]
	if os.path.exists(label_map_dir)==True:
		with open(label_map_dir) as f:
			contents = f.readlines()
			for c in contents:
				c=c.split()[0]
				label_map.append(c)
	num_classes=len(label_map)
	return label_map,num_classes
	
def create_inferece_model():
	global vis_thresh
	path='./export/'
	class_name,num_classes=read_map(path)	
	opt = opts()
	args = opt.parser.parse_args()
	args.model_name="centernet"
	# args.backbone="hardnet85"
	args.backbone="dla34"	
	args.task='ctdetp'
	args.mean= [0, 0, 0]
	args.std=[1, 1, 1]
	args.phase="test"
	# args.pretrained=False
	args.resume=True
	args.class_name=class_name
	args.vis_thresh=vis_thresh
	args=opt.parse_dict(args)
	Dataset = get_dataset(args.model_name, args.task)
	Dataset.default_resolution=[512,512]
	Dataset.num_classes=num_classes	
	args = opt.update_dataset_info_and_set_heads(args, Dataset)	
	# args.load_model=os.path.join(path,'hardnet.pth')	
	args.load_model=os.path.join(path,'model_last.pth') #aug eval best 
	# args.load_model=os.path.join(path,'model_1000.pth')	# aug
	# args.load_model=os.path.join(path,'model_400.pth')	#not aug 	
	opt=args
	os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
	Detector = detector_factory[opt.task]
	detector = Detector(opt) 
	return detector
	


			
def run_video():
	global zoom_factor
	global vis_thresh
	global isRun
	model=create_inferece_model()
	opt=model.opt
	opt.demo="webcam"
	
	times = []
	is_video=False
	vid_h=0
	vid_w=0

	cam = cv2.VideoCapture(cv2.CAP_DSHOW)
	cam.set(cv2.CAP_PROP_AUTOFOCUS, opt.cam_focus)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.cam_res[1])
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, opt.cam_res[0])
	print("{} mode".format(opt.demo))
	
	debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),theme=opt.debugger_theme,opt=opt)

	avg_stats = {'tot': AverageMeter(), 'load': AverageMeter(),'pre':  AverageMeter(), 'net':  AverageMeter(), 'dec':  AverageMeter(),
			'post':  AverageMeter(), 'merge':  AverageMeter()}
	model.pause = False
	while isRun:
		_, img = cam.read()
		if(img is not None):
			h,w,_=img.shape
			img=zoom_io(img,zoom_factor)
			img=cv2.resize(img,(w,h))	
			ret = model.run(img)
			model.opt.vis_thresh=vis_thresh
			model.show_results(debugger,img,ret['results'])
			for i, v in debugger.imgs.items():
				cv2.imshow('{}'.format(i),v)

		if cv2.waitKey(1) == 27:
			break
	cam.release()
	print("close")



if __name__ == '__main__':
	th = threading.Thread(target=run_video, args=())
	th.start()
	run_input()	
