import pdb

import os
import pickle
import cv2
import numpy as np
import torch
import torch.utils.data as data

from torch.utils.data import DataLoader

from pycocotools.coco import COCO


class COCO_dataloader:
	def __init__(self, root, split,target_transform=None):
		
		self.resize = [416,416]
		self.means = (103.94, 116.78, 123.68)

		self.root = root
		self.target_transform = target_transform
		self.anno_file = self.root + 'annotations/instances_' + split + '2017.json'
		_COCO = COCO(self.anno_file)

		Categories = _COCO.loadCats(_COCO.getCatIds())

		self._classes = tuple(['__background__'] + [c['name'] for c in Categories])
		self.num_classes = len(self._classes)
		self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
		self._class_to_coco_cat_id = dict(zip([c['name'] for c in Categories],
                                                  _COCO.getCatIds()))

		indexes = _COCO.getImgIds()
		self.image_indexes = indexes
		self.ids= self.load_coco_img_path(split,indexes)
		self.annotations = self._load_coco_annotations(split,indexes,_COCO)

		#pdb.set_trace()
	
	def load_coco_img_path(self,split,indexes):
		coco_file = os.path.join(self.root,'images_train.pkl')
		if os.path.exists(coco_file):
			with open(coco_file, 'rb') as fid:
				gt_roib = pickle.load(fid)
			return gt_roib
		img_path = [self.image_path_from_index(split,index) for index in indexes]
		with open(coco_file,'wb') as fid:
			pickle.dump(img_path,fid,pickle.HIGHEST_PROTOCOL)
		return img_path

	def image_path_from_index(self,name,index):
		file_name = str(index).zfill(12)+'.jpg'
		image_path = os.path.join(self.root,name+'2017',file_name)
		assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
		return image_path

	def _load_coco_annotations(self,split,indexes,_COCO):
		coco_file = os.path.join(self.root,'annotations_train.pkl')
		if os.path.exists(coco_file):
			with open(coco_file, 'rb') as fid:
				gt_roib = pickle.load(fid)
			return gt_roib
		gt_roib = [self._annotation_from_index(index,_COCO) for index in indexes]
		with open(coco_file,'wb') as fid:
			pickle.dump(gt_roib,fid,pickle.HIGHEST_PROTOCOL)
		return gt_roib

	def _annotation_from_index(self,index,_COCO):
		im_ann = _COCO.loadImgs(index)[0]
		width = im_ann['width']
		height = im_ann['height']

		annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
		objs = _COCO.loadAnns(annIds)
		#pdb.set_trace()
		valid_objs = []
		for obj in objs:
			x1 = np.max((0, obj['bbox'][0]))
			y1 = np.max((0, obj['bbox'][1]))
			x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
			y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
			if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
				obj['clean_bbox'] = [x1, y1, x2, y2]
				valid_objs.append(obj)
		objs = valid_objs
		num_objs = len(objs)

		#res = np.zeros((num_objs, 5))
		#pdb.set_trace()
		# Lookup table to map from COCO category ids to our internal class
		# indices
		coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
		                                  self._class_to_ind[cls])
		                                 for cls in self._classes[1:]])
		output = []
		for ix, obj in enumerate(objs):
			cls = coco_cat_id_to_class_ind[obj['category_id']]
			if cls in [1,2,3]:
				output = output + [obj['clean_bbox']+[cls]]
		if len(output)>0:
			output = np.stack(output,axis=0)	
		else:
			return np.array([])	
		
		return output

	def transform(self,image,size,means):
		image = cv2.resize(image,(size[0],size[1]),interpolation=cv2.INTER_CUBIC)
		image = image.astype(np.float32)
		image -= means
		return image.transpose(2,0,1)

	def __getitem__(self,index):
		img_id = self.ids[index]
		target = self.annotations[index]

		img = cv2.imread(img_id,cv2.IMREAD_COLOR)
		height,width,_ = img.shape

		if len(target) == 0:
			targets = np.zeros((1,5))
			img = self.transform(img,self.resize,self.means)
			return torch.from_numpy(img),targets



		if self.target_transform is not None:
			target = self.target_transform(target)

		boxes = target[:,:-1].copy()
		labels = target[:,-1].copy()		

		#img,target
		height, width,_ = img.shape
		img = self.transform(img,self.resize,self.means)
		boxes[:, 0::2] /= width
		boxes[:, 1::2] /= height

		labels = np.expand_dims(labels,1)
		targets = np.hstack((boxes,labels))

		return torch.from_numpy(img),targets

	def __len__(self):
		return len(self.image_indexes)


def detection_collate(batch):
	"""
	Custom collate fn for dealing with batches of images that have a different
	number of associated object annotations (bounding boxes).
	Arguments:
	    batch: (tuple) A tuple of tensor images and lists of annotations
	Return:
		A tuple containing:
		1) (tensor) batch of images stacked on their 0 dim
		2) (list of tensors) annotations for a given image are stacked on 0 dim
	"""
	targets = []
	imgs = []
	for _, sample in enumerate(batch):
		for _, tup in enumerate(sample):
			if torch.is_tensor(tup):
				imgs.append(tup)
			elif isinstance(tup, type(np.empty(0))):
				annos = torch.from_numpy(tup).float()
				targets.append(annos)

	return (torch.stack(imgs, 0), targets)


if __name__ == "__main__":	



	dataiterator = COCO_dataloader('/data/Docker_Data/COCO/','train')
	loader = DataLoader(dataiterator, batch_size = 12,num_workers=16,shuffle=True,collate_fn=detection_collate, pin_memory=True)
	batch_iterator = iter(loader)
	
	
	for image,target in batch_iterator:
		image = image[1]
		
		_,height,width = image.shape
		x1 = (target[1][0][0::2]*width).int().numpy()
		y1 = (target[1][0][1::2]*height).int().numpy()
		#image = cv2.rectangle(image, start_point, end_point, color, thickness) 
		
		mean = (image.numpy()[0].min(),image.numpy()[1].min(),image.numpy()[2].min())
		im = image.permute(1,2,0).numpy()-mean
		im = cv2.rectangle(im.astype(np.uint8), (x1[0], y1[0]), (x1[1], y1[1]), (0, 255, 0), 2)
		
		cv2.imshow('image',im)
		cv2.waitKey(0)
		pdb.set_trace()
