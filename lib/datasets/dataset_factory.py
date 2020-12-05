from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .sampler.centernetdet_sampler import CenternetDetSampler
from .dataset.centernet_det import CenternetDetDataset

from .sampler.centernetdet_patch_sampler import CenternetDetPatchSampler
from .dataset.centernet_det_patch import CenternetDetPatchDataset



import os

def write_map(logger,class_name):
	f = open(os.path.join(logger.log_dir,"map.txt"),"w")
	for key in class_name:
		label="{}\n".format(key)
		f.write(label)
	f.close()	
	
def read_patch(data_dir):
	
	back_path=os.path.join(data_dir,"background")
	patch_path=os.path.join(data_dir,"patch")
	negative_path=os.path.join(data_dir,"negative")

	negative_label=[]
	for root, dirnames, filenames in os.walk(negative_path):
		for filename in filenames:
			negative_label.append(os.path.join(root,filename))

	back_label=[]
	for root, dirnames, filenames in os.walk(back_path):
		for filename in filenames:
			back_label.append(os.path.join(root,filename))

	label_map=[]
	patch_label=[]			
	for root, dirnames, filenames in os.walk(patch_path):
		
		if(len(filenames)<=0):
			continue
		label=os.path.basename(root)
		label_map.append(label)
		patch_list=[]
		for filename in filenames:
			patch_list.append(os.path.join(root,filename))
		patch_label.append({label:patch_list})
			
	return back_label,patch_label,negative_label,label_map

def create_label(origin_labels,path,task):
	if task=="cla" or task =="seg":
		f = open(path,"w")
		for label in origin_labels:
			dir=label['fname']
			label=label['label']
			contents="{}|{}\n".format(dir,label)
			f.write(contents)		
		f.close()
	elif task=="ctdet":
		f = open(path,"w")
		for label in origin_labels:
			dir=label['fname']
			str_label=""
			for anno in label['label']:
				bbox=anno[0:4]
				cate=anno[4]
				out_str=[','.join([str(b) for b in bbox])+","+str(cate)+"^"]
				str_label+=out_str[0]
			
			contents="{}|{}\n".format(dir,str_label)
			f.write(contents)		
		f.close()
	elif task=="mpp":
		f = open(path,"w")
		for label in origin_labels:
			dir=label['fname']
			str_label=""
			for anno in label['label']:
				bbox=anno[0:4]
				cate=anno[4]
				keypoints=anno[4:]
				out_str=[','.join([str(b) for b in bbox])+","+str(cate)+","+','.join([str(kp) for kp in keypoints])+"^"]
				str_label+=out_str[0]
			
			contents="{}|{}\n".format(dir,str_label)
			f.write(contents)		
		f.close()
		
		
		
def count_category(origin_labels,task):
	class_to_count={}
	if task=="cla":
		for i in range(len(origin_labels)):
			label=origin_labels[i]['label']
			if not label in class_to_count.keys():
				class_to_count[label]=0
				class_to_count[label]+=1
			else:
				class_to_count[label]+=1
	elif task=="ctdet" or task=="mpp":				
		for i in range(len(origin_labels)):
			for anno in origin_labels[i]['label']:
				label=anno[4]
				if not label in class_to_count.keys():
					class_to_count[label]=0
					class_to_count[label]+=1
				else:
					class_to_count[label]+=1				
	def sort_key(x):
		return x[0]
	class_to_count=dict(sorted(class_to_count.items(),key=sort_key,reverse=True))
	return class_to_count
	

def selective_folding(gt_labels,class_name,logger,task):


	f = open(os.path.join(logger.log_dir,"map.txt"),"w")
	for key in class_name:
		label="{}\n".format(key)
		f.write(label)
	f.close()	

	dataset_count=int(len(gt_labels)*0.2)

	train_labels=gt_labels[dataset_count+dataset_count:]
	valid_labels=gt_labels[dataset_count:dataset_count+dataset_count]
	test_labels=gt_labels[0:dataset_count]
	
	
	create_label(train_labels,os.path.join(logger.log_dir,"train_labels.txt"),task)
	create_label(valid_labels,os.path.join(logger.log_dir,"valid_labels.txt"),task)
	create_label(test_labels,os.path.join(logger.log_dir,"test_labels.txt"),task)
	
	if(task=='seg'):
		logger.write("==> data folding:\n")			
		logger.write("train_dataset_sum:{}\n".format(len(train_labels)))
		logger.write("valid_dataset_sum:{}\n".format(len(valid_labels)))	
		logger.write("test_dataset_sum:{}\n".format(len(test_labels)))
		return train_labels,valid_labels,test_labels

	
	train_class_to_count=count_category(train_labels,task)
	valid_class_to_count=count_category(valid_labels,task)
	test_class_to_count=count_category(test_labels,task)

	logger.write("==> data folding:\n")			
	logger.write("train_dataset_to_class_count:{}, sum:{}\n".format(train_class_to_count,len(train_labels)))
	logger.write("valid_dataset_to_class_count:{}, sum:{}\n".format(valid_class_to_count,len(valid_labels)))	
	logger.write("test_dataset_to_class_count:{}, sum:{}\n".format(test_class_to_count,len(test_labels)))
	
	return train_labels,valid_labels,test_labels


def read_label_txt(data_dir,anno_dir,task):
	gt_labels=[]
	if task=="cla" or  task=="seg":
		with open(anno_dir) as f:
			contents = f.readlines()
			for c in contents:
				c=c.split('\n')[0]
				c=c.split('|')
				fname=os.path.join(c[0])
				label=c[1]
				gt_labels.append({'fname': fname,'label': label}) 
	elif task=="det":
		with open(anno_dir) as f:
			contents = f.readlines()
			for c in contents:
				c=c.split('\n')[0]
				c=c.split('|')
				fname=os.path.join(c[0])
				raw_labels=c[1].split("^")[:-1]
				label=[]
				for raw_label in raw_labels: 
					anno=raw_label.split(",")
					anno[0]=int(anno[0])
					anno[1]=int(anno[1])
					anno[2]=int(anno[2])					
					anno[3]=int(anno[3])
					label.append(anno)
				gt_labels.append({'fname': fname,'label':label })
	elif task=="mpp":
		with open(anno_dir) as f:
			contents = f.readlines()
			for c in contents:
				c=c.split('\n')[0]
				c=c.split('|')
				fname=os.path.join(c[0])
				raw_labels=c[1].split("^")[:-1]
				label=[]
				for raw_label in raw_labels: 
					anno=raw_label.split(",")
					anno[0]=int(anno[0])
					anno[1]=int(anno[1])
					anno[2]=int(anno[2])					
					anno[3]=int(anno[3])
					for idx in range(5,len(anno)):
						anno[idx]=int(anno[idx])
					label.append(anno)
				gt_labels.append({'fname': fname,'label':label })				
	return gt_labels

def write_label_ts3(path,label,x_center,y_center,w,h,r=0,id=-1):
	import configparser
	config = configparser.ConfigParser()
	config.read(path)
	label=label.upper()
	delimiter=","
	key=str(x_center)+delimiter+str(y_center)+delimiter+str(w)+delimiter+str(h)+delimiter+str(r)+delimiter+str(id)
	if(config.has_section(label)==False):
		config.add_section(label)
	config.set(label, key, 'detection')
	with open(path, 'w') as f:
		  config.write(f)


def read_label_ts3(data_dir):
	gt_labels=[]
	file_dir = os.listdir(data_dir)
	file_list = [os.path.join(data_dir,file) for file in file_dir if file.endswith(".ts3")]
	image_format_map={}
	for file in file_dir:
		if not file.endswith(".ts3"):
			map=file.split(".")
			image_format_map[map[0]]=map[1]
	import configparser
	config = configparser.ConfigParser()	
	config.read(file_list)
	categories=config.sections()
	categories.sort()
	for file in file_list:
		config = configparser.ConfigParser()	
		config.read(file)	
		file_categories=config.sections()
		label=[]
		for cate in file_categories:
			for contents, task in config.items(cate):
				anno=contents.split(",")
				anno=anno[:4]
				anno.append(cate)
				label.append(anno)
		file_name=os.path.basename(file).split(".")[0]
		fname=file_name+"."+image_format_map[file_name]
		gt_labels.append({'fname': fname,'label':label })	
	
	return gt_labels,categories

def read_label(data_dir,anno_dir,task):		
	return read_label_ts3(data_dir)
			 
def read_data(data_dir,resume_labels,task):
	if resume_labels is False:
		gt_labels,label_map=read_label_ts3(data_dir)
		return gt_labels,label_map
	else:
		label_map_dir = os.path.join(data_dir, "map.txt")
		label_map=[]
		if os.path.exists(label_map_dir)==True:
			with open(label_map_dir) as f:
				contents = f.readlines()
				for c in contents:
					c=c.split('\n')[0]
					label_map.append(c)		
		anno_dir = os.path.join(data_dir,"train_labels.txt")
		train_labels=read_label_txt(data_dir,anno_dir,task)
		anno_dir = os.path.join(data_dir,"valid_labels.txt")		
		valid_labels=read_label_txt(data_dir,anno_dir,task)	
		anno_dir = os.path.join(data_dir,"test_labels.txt")
		test_labels=read_label_txt(data_dir,anno_dir,task)
		return train_labels,valid_labels,test_labels,label_map

class DetDataset(CenternetDetDataset,CenternetDetSampler):
	pass
class DetPatchDataset(CenternetDetPatchDataset,CenternetDetPatchSampler):
	pass	

def get_dataset(dataset, task):
	if(task=="ctdet"):
		#if(dataset=="centernet"):
		return DetDataset
	else:
		#if(dataset=="centernet"):
		return DetPatchDataset
  
