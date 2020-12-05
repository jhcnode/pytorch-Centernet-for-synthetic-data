import numpy as np
from .ap_accumulator import APAccumulator
from .utils.bbox import jaccard
import math



class DetectionMAP:
	def __init__(self, n_class, pr_samples=101, overlap_threshold=0.5):
		self.n_class = n_class
		self.overlap_threshold = overlap_threshold
		self.pr_scale = np.linspace(0, 1.0, pr_samples)
		self.total_accumulators = []
		self.reset_accumulators()

	def reset_accumulators(self):
		self.total_accumulators = []
		for i in range(len(self.pr_scale)):
			class_accumulators = []
			for j in range(self.n_class):
				class_accumulators.append(APAccumulator())
			self.total_accumulators.append(class_accumulators)
			

	def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
		if pred_bb.ndim == 1:
			pred_bb = np.repeat(pred_bb[:, np.newaxis], 4, axis=1)
		IoUmask = None
		if len(pred_bb) > 0:
			IoUmask = self.compute_IoU_mask(pred_bb, gt_bb, self.overlap_threshold)
		for accumulators, r in zip(self.total_accumulators, self.pr_scale):
			self.evaluate_(IoUmask, accumulators, pred_classes, pred_conf, gt_classes, r)

	@staticmethod
	def evaluate_(IoUmask, accumulators, pred_classes, pred_conf, gt_classes, confidence_threshold):
		pred_classes = pred_classes.astype(np.int)
		gt_classes = gt_classes.astype(np.int)

		for i, acc in enumerate(accumulators):
			gt_number = np.sum(gt_classes == i)
			pred_mask = np.logical_and(pred_classes == i, pred_conf >= confidence_threshold)
			pred_number = np.sum(pred_mask)
			if pred_number == 0:
				acc.inc_not_predicted(gt_number)
				continue

			IoU1 = IoUmask[pred_mask, :]
			mask = IoU1[:, gt_classes == i]

			tp = DetectionMAP.compute_true_positive(mask)
			fp = pred_number - tp
			fn = gt_number - tp
			acc.inc_good_prediction(tp)
			acc.inc_not_predicted(fn)
			acc.inc_bad_prediction(fp)

	@staticmethod
	def compute_IoU_mask(prediction, gt, overlap_threshold):
		IoU = jaccard(prediction, gt)
		for i in range(len(prediction)):
			maxj = IoU[i, :].argmax()
			IoU[i, :maxj] = 0
			IoU[i, (maxj + 1):] = 0
		return IoU >= overlap_threshold

	@staticmethod
	def compute_true_positive(mask):
		# sum all gt with prediction of its class
		return np.sum(mask.any(axis=0))

	def compute_ap(self, precisions, recalls):
		previous_recall = 0
		average_precision = 0
		for precision, recall in zip(precisions[::-1], recalls[::-1]):
			average_precision += precision * (recall - previous_recall)
			previous_recall = recall
		return average_precision


	def compute_precision_recall_(self, class_index, interpolated=True):
		precisions = []
		recalls = []
		for acc in self.total_accumulators:
			precisions.append(acc[class_index].precision)
			recalls.append(acc[class_index].recall)

		if interpolated:
			interpolated_precision = []
			for precision in precisions:
				last_max = 0
				if interpolated_precision:
					last_max = max(interpolated_precision)
				interpolated_precision.append(max(precision, last_max))
			precisions = interpolated_precision
		return precisions, recalls
		

	def plot(self, interpolated=True, class_names=None):
		mean_average_precision=[]
		for i in range(self.n_class):
			precisions, recalls = self.compute_precision_recall_(i, interpolated)
			average_precision = self.compute_ap(precisions, recalls)
			class_name = class_names[i] if class_names else "Class {}".format(i)
			mean_average_precision.append(average_precision)
			
		return sum(mean_average_precision)/len(mean_average_precision)
		# result=""
		# for i, ap in enumerate(mean_average_precision):
			# result+="class {} AUC : {:0.2f} ".format(i,ap)
			
		# print("{}".format(result))
		# print("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))
