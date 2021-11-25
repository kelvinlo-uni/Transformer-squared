import numpy as np
import json
import tensorflow as tf
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, TFAutoModel
import nltk
from nltk.tokenize import sent_tokenize
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

nltk.download('punkt')


MASK = '[mask]'
PAD = '[pad]'
UNK = '[unk]'
special_tokens = [MASK,PAD,UNK]

corpus = 'en_city' # en_city/en_disease/de_city/de_disease

def assign_GPU_pt(Tokenizer_output):
	output = {}
	for i in Tokenizer_output:
		output[i] = Tokenizer_output[i].to('cuda:0')
	return output

def prep_data(model = 'BERT', corpus = 'en_city', max_sent = 512, test_d_size = None, is_tf = True,train=True,validation=True,test=True):
	original_path = 'data/wikisection/'
	processed_path = 'processed_data/' + model + '/'
	
	Path(processed_path).mkdir(parents=True, exist_ok=True)
	if corpus == 'en_city':
		original_path = original_path + 'wikisection_en_city_'
		processed_path = processed_path + 'wikisection_en_city_'
	elif corpus == 'en_disease':
		original_path = original_path + 'wikisection_en_disease_'
		processed_path = processed_path + 'wikisection_en_disease_'
	elif corpus == 'de_city':
		original_path = original_path + 'wikisection_de_city_'
		processed_path = processed_path + 'wikisection_de_city_'
	elif corpus == 'de_disease':
		original_path = original_path + 'wikisection_de_disease_'
		processed_path = processed_path + 'wikisection_de_disease_'
	else:
		return None
	filenames = []
	if train:
		filenames.append('train.json')
	if validation:
		filenames.append('validation.json')
	if test:
		filenames.append('test.json')
		
	model_dict = {
		'tf': {
			'ALBERT': 'albert-base-v2',
			'BERT': 'bert-base-cased',
			'BERTL': 'bert-large-cased',
			'BLUEBERT': 'ttumyche/bluebert',
			'DEBERT': 'bert-base-german-cased',
			'XLNET': 'xlnet-base-cased'
		},
		'pytorch': {
			'BIOCLINICALBERT': 'emilyalsentzer/Bio_ClinicalBERT',
			'BIOMED_ROBERTA': 'allenai/biomed_roberta_base',
		}
		
	}
	
	assert (is_tf and model in model_dict['tf']) or (not is_tf and model in model_dict['pytorch']), 'Model found for ' + model + (is_tf and ' on tf' or ' on pytorch')
	model_name = is_tf and model_dict['tf'][model] or model_dict['pytorch'][model]
	
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	
	if is_tf:
		Model = TFAutoModel.from_pretrained(model_name)
	else:
		Model = AutoModel.from_pretrained(model_name, return_dict=True)
		

	labels = set()

	for fn in filenames:
		print('Now processing: ' + str(fn))
		with open(original_path + fn, 'r+') as f:
			data = json.load(f)

		# test
		if test_d_size and test_d_size > 0:
			data = data[:test_d_size]

		processed_data = []
		for data_idx in tqdm(range(0, len(data))):
			data_item = data[data_idx]
			t = data_item['text']
			a = data_item['annotations']
			sections = []
			for ai in a:
				si = {}
				si['text'] = t[ai['begin']: ai['begin'] + ai['length']]
				si['label'] = ai['sectionLabel']
				sections.append(si)
				labels.add(ai['sectionLabel'])
			sents = []
			current_section = ''
			for si in sections:
				st = sent_tokenize(si['text'])
				for sti in st:
					if len(sents) >= max_sent:
						# truncate
						break
					begin = 0 if (current_section == si['label']) else 1
					current_section = si['label']
					if is_tf:
						if model == 'XLNET':
							inputs = tokenizer(sti, return_tensors="tf",padding=True,truncation=True,max_length=512)
							outputs = Model(inputs)
							# each sentence dictionary has 'cls', 'label' and 'begin' keys, the values are the sentence embedding, the topic label and begin sentence indicator
							sents.append({'cls': str(outputs[0][:,-1,:].numpy().tolist()[0]), 'label': si['label'], 'begin': begin}) 
						else:
							inputs = tokenizer(sti, return_tensors="tf",padding=True,truncation=True,max_length=512)
							outputs = Model(inputs)
							sents.append({'cls': str(outputs[0][:,0,:].numpy().tolist()[0]), 'label': si['label'], 'begin': begin})
					else:
						if torch.cuda.is_available():  
							dev = "cuda:0" 
							inputs = assign_GPU_pt(tokenizer(sti, return_tensors="pt",padding=True,truncation=True,max_length=512))
							Model = Model.to(dev)
							outputs = Model(**inputs)
							sents.append({'cls': str(outputs.last_hidden_state[:,0,:].cpu().detach().numpy().tolist()[0]), 'label': si['label'], 'begin': begin})
						else:  
							dev = "cpu"
							inputs = tokenizer(sti, return_tensors="pt",padding=True,truncation=True,max_length=512)
							outputs = Model(**inputs)
							sents.append({'cls': str(outputs.last_hidden_state[:,0,:].detach().numpy().tolist()[0]), 'label': si['label'], 'begin': begin})
			doc_len = len(sents)
			#for i in range(0,(max_sent-len(sents))):
				# padding
				#sents.append({'cls': str(np.zeros(768).tolist()), 'label': '[pad]', 'begin': '[pad]'})
			processed_data.append({'sent':sents, 'doc_len': doc_len})

		with open(processed_path + fn, 'w+') as f:
			json.dump(processed_data,f)

	print('Done!')
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help="BERT/BERTL/BLUEBERT/BIOCLINICALBERT/DEBERT")
	parser.add_argument('--corpus', help="en_city/en_disease/de_city/de_disease")
	parser.add_argument('--max_sent', help="maximum number of sentence in a document", default='512')
	parser.add_argument('--test_d_size', help="for testing only: number of data used create output", default='None')
	parser.add_argument('--tf', help="Tensorflow if true; Pytorch if false (T/F)", default='T')
	parser.add_argument('--train',default='T')
	parser.add_argument('--validation',default='T')
	parser.add_argument('--test',default='T')
	
	args = parser.parse_args()
		
		
	assert args.max_sent.isdigit(), 'max_sent should be an integer'
	max_sent = int(args.max_sent)
	
	assert args.test_d_size.isdigit() or args.test_d_size == 'None', 'test_d_size should be an integer or None'
	
	if args.test_d_size.isdigit():
		test_d_size = int(args.test_d_size)
	else:
		test_d_size = None
	
		
	prep_data(args.model,args.corpus,max_sent,test_d_size,args.tf.upper() == 'T',args.train.upper()=='T',args.validation.upper()=='T',args.test.upper()=='T')
	
