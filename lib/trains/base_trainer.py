from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModelWithLoss(torch.nn.Module):
	def __init__(self, model, loss):
		super(ModelWithLoss, self).__init__()
		self.model = model
		self.loss = loss

	def forward(self, batch):
		outputs = self.model(batch['input'])
		loss, loss_stats = self.loss(outputs, batch)
		return outputs[-1], loss, loss_stats

class BaseTrainer(object):
	def __init__(
	self, opt, model, optimizer=None):
		self.opt = opt
		self.optimizer = optimizer
		self.loss_stats, self.loss = self._get_losses(opt)
		self.eval_stats=self._get_evals(opt)
		self.model=model
		self.model_with_loss = ModelWithLoss(self.model, self.loss)

	def set_device(self, gpus, chunk_sizes, device):
		if len(gpus) > 1:
			self.model_with_loss = DataParallel(
			self.model_with_loss, device_ids=gpus, 
			chunk_sizes=chunk_sizes).to(device)
		else:
			self.model_with_loss = self.model_with_loss.to(device)

		for state in self.optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device=device, non_blocking=True)

	def run_epoch(self, phase, epoch, data_loader):
		
		model_with_loss = self.model_with_loss
		
		if phase == 'train':
			model_with_loss.train()
		else:
			if len(self.opt.gpus) > 1:
				model_with_loss = self.model_with_loss.module
			model_with_loss.eval()
			torch.cuda.empty_cache()


		opt = self.opt
		results = {}
		data_time, batch_time = AverageMeter(), AverageMeter()
		avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
		avg_eval_stats = {l: AverageMeter() for l in self.eval_stats}
		
		if phase == 'train':
			num_iters = len(data_loader.dataset)//opt.batch_size
		else:
			num_iters = len(data_loader.dataset)		
		
		bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
			
		
		dataiterator = iter(data_loader)
		end = time.time()
		
		for iter_id in range(num_iters+1):
			# if iter_id%100 == 0 and  phase == 'train':
				# save_model(os.path.join(opt.save_dir, 'model_epoch_{}_iter_{}.pth'.format(epoch-1,iter_id)),epoch-1, self.model, self.optimizer)
			
			self.optimizer.zero_grad()
			batch_loss_stats = {l: 0 for l in self.loss_stats}
			batch_eval_stats = {l: 0 for l in self.eval_stats}
			if phase == 'train':
				subdivision=opt.subdivision
			else:
				subdivision=1
				

			for sub_iter in range(subdivision):
				end = time.time()
				try:
					batch = next(dataiterator) 
				except StopIteration:
					dataiterator = iter(data_loader)
					batch = next(dataiterator) 
				data_time.update(time.time() - end)

				end = time.time()
				for k in batch:
					if k != 'meta':
						batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
				output, loss, loss_stats = model_with_loss(batch)
				loss = loss.mean()
				batch_time.update(time.time() - end)

				end = time.time()
				for l in batch_loss_stats:
					batch_loss_stats[l]+=loss_stats[l].item()/subdivision
					
									
				if phase == 'train':
					loss.backward()
				else:
					pass
					# test_stats=self._get_result(batch,output)
					# for l in batch_eval_stats:
						# batch_eval_stats[l] +=test_stats[l]/subdivision

						
				if opt.debug > 1 and  phase == 'test' :
					self.debug(batch, output, sub_iter)
					
			if phase == 'train':
				self.optimizer.step()


				
			Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
			epoch, iter_id, num_iters, phase=phase,
			total=bar.elapsed_td, eta=bar.eta_td)
			if phase=='test':
				batch_size=1
			else:
				batch_size=opt.batch_size
				
			for l in avg_loss_stats:
				avg_loss_stats[l].update(
				batch_loss_stats[l],batch_size)
				Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
				
			if not opt.hide_data_time:
				Bar.suffix = Bar.suffix + '|Data {dt.val:.7f}s({dt.avg:.3f}s) ' \
				'|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
				
			if phase=='test':
				for l in avg_eval_stats:
					avg_eval_stats[l].update(
					batch_eval_stats[l], batch_size)			
					Bar.suffix =  Bar.suffix +'|{} {:.4f} '.format(l,avg_eval_stats[l].avg)
							
			if opt.print_iter > 0:
				if iter_id % opt.print_iter == 0:
					print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
			else:
				bar.next()	
								
				
		bar.finish()
		ret = {k: v.avg for k, v in avg_loss_stats.items()}		
		ret['time'] = bar.elapsed_td.total_seconds() / 60.
		if phase == 'test':
			for l in avg_eval_stats:
				ret[l]=avg_eval_stats[l].avg	
		return ret,results

	def debug(self, batch, output, iter_id):
		raise NotImplementedError

	def save_result(self, output, batch, results):
		raise NotImplementedError

	def _get_losses(self, opt):
		raise NotImplementedError
		
	def _get_evals(self, opt):
		raise NotImplementedError

	def test(self, epoch, data_loader):
		return self.run_epoch('test', epoch, data_loader)

	def train(self, epoch, data_loader):
		return self.run_epoch('train', epoch, data_loader)