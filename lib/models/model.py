from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import os


from .networks.detector.centernet.msra_resnet import get_pose_net as get_msra_resnet
from .networks.detector.centernet.dlav0 import get_pose_net as get_dlav0
from .networks.detector.centernet.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.detector.centernet.resnet_dcn import get_pose_net as get_resnet_dcn
from .networks.detector.centernet.large_hourglass import get_large_hourglass_net
from .networks.detector.centernet.hardnet import get_pose_net as get_hardnet


_model_factory = {
	'ctdet':{
		'centernet':{
			'resnet': get_msra_resnet, # default Resnet with deconv
			'dlav': get_dlav0, # default DLAup
			'dla': get_dla_dcn,
			'resdcn': get_resnet_dcn,
			'hourglass': get_large_hourglass_net
		}
	},
	'ctdetp':{
		'centernet':{
			'resnet': get_msra_resnet, # default Resnet with deconv
			'dlav': get_dlav0, # default DLAup
			'dla': get_dla_dcn,
			'resdcn': get_resnet_dcn,
			'hourglass': get_large_hourglass_net,
			'hardnet': get_hardnet
			
		}
	}
		


}

def create_model(**kwargs):
	opt=kwargs['opt']
	if opt.task=='cla':
		return create_classifier(kwargs['model_name'],kwargs['opt'])	
	elif opt.task=='ctdet' or opt.task=='ctdetp' : 
		return create_detector(kwargs['model_name'],kwargs['backbone'],kwargs['opt'])	
	elif opt.task=='seg':
		return create_segmentator(kwargs['model_name'],kwargs['backbone'],kwargs['opt'])		
	else:
		assert "형식 모델 없음"
	
def create_detector(detector,backbone,opt):
	backbone=backbone.lower()
	detector=detector.lower()
	
	num_layers = int(''.join(x for x in backbone if x.isdigit()))
	backbone=''.join(x for x in backbone if not x.isdigit())
	get_model = _model_factory[opt.task][detector][backbone]
	model = get_model(num_layers=num_layers, heads=opt.heads, head_conv=opt.head_conv)
	return model		
		
def create_classifier(classfier,opt):
	get_model = _model_factory[opt.task][classfier]
	model = get_model(num_classes=opt.num_classes)
	return model
	
def create_segmentator(detector,backbone,opt):
	backbone=backbone.lower()
	segmentator=detector.lower()
	get_model = _model_factory[opt.task][segmentator][backbone]
	model = get_model(num_classes=opt.num_classes)
	return model

	
def load_model(model, model_path, optimizer=None, resume=False, 
				lr=None, lr_step=None):
	start_epoch = 0
	checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
	print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
	state_dict_ = checkpoint['state_dict']
	state_dict = {}

	# convert data_parallal to model
	for k in state_dict_:
		# print(k)
		if k.startswith('module') and not k.startswith('module_list'):
			state_dict[k[7:]] = state_dict_[k]
		else:
			state_dict[k] = state_dict_[k]
		model_state_dict = model.state_dict()

	# check loaded parameters and created model parameters
	for k in state_dict:
		if k in model_state_dict:
			if state_dict[k].shape != model_state_dict[k].shape:
				print('Skip loading parameter {}, required shape{}, '\
				'loaded shape{}.'.format(
				k, model_state_dict[k].shape, state_dict[k].shape))
				state_dict[k] = model_state_dict[k]
		else:
			print('Drop parameter {}.'.format(k))
			
			
	for k in model_state_dict:
		if not (k in state_dict):
			print('No param {}.'.format(k))
			state_dict[k] = model_state_dict[k]
		model.load_state_dict(state_dict, strict=False)

	# resume optimizer parameters
	if optimizer is not None and resume is True:
		if 'optimizer' in checkpoint:
			optimizer.load_state_dict(checkpoint['optimizer'])
			start_epoch = checkpoint['epoch']
			start_lr = lr
			for step in lr_step:
				if start_epoch >= step:
					start_lr *= 0.1
				
			for param_group in optimizer.param_groups:
				param_group['lr'] = start_lr
			print('Resumed optimizer with start lr', start_lr)
		else:
			print('No optimizer parameters in checkpoint.')
		  
	
	if optimizer is not None:
		return model, optimizer, start_epoch
	else:
		return model

def save_model(path, epoch, model, optimizer=None, valid_acc=None):
	if isinstance(model, torch.nn.DataParallel):
		state_dict = model.module.state_dict()
	else:
		state_dict = model.state_dict()
	data = {'epoch': epoch,
          'state_dict': state_dict,
          'valid_acc': valid_acc}
	if not (optimizer is None):
		data['optimizer'] = optimizer.state_dict()
	torch.save(data, path)

