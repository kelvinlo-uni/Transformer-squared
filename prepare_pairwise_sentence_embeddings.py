"""
Create pairwise sentence ebmeddings.
"""

import numpy as np
import json
import tensorflow as tf
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, TFAutoModel, TFBertForNextSentencePrediction, BertForNextSentencePrediction
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
		processed_path = processed_path + 'wikisection_en_city_pairwise_'
	elif corpus == 'en_disease':
		original_path = original_path + 'wikisection_en_disease_'
		processed_path = processed_path + 'wikisection_en_disease_pairwise_'
	elif corpus == 'de_city':
		original_path = original_path + 'wikisection_de_city_'
		processed_path = processed_path + 'wikisection_de_city_pairwise_'
	elif corpus == 'de_disease':
		original_path = original_path + 'wikisection_de_disease_'
		processed_path = processed_path + 'wikisection_de_disease_pairwise_'
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
	# if model == 'BERT':
		# model_name = 'bert-base-cased'
	# elif model == 'BERTL':
		# model_name = 'bert-large-cased'
	# elif model == 'BLUEBERT':
		# model_name = 'ttumyche/bluebert'
	# elif model == 'BIOCLINICALBERT':
		# model_name = 'emilyalsentzer/Bio_ClinicalBERT'
		# assert not is_tf, 'Not TF model found for ' + model_name
	# elif model == 'DEBERT':
		# model_name = 'bert-base-german-cased'
	# elif model == 'XLNET':
		# model_name = 'xlnet-base-cased'
	# elif model == 'ALBERT':
		# model_name = 'albert-base-v2'
	# elif model == 'ROBERTA':
		# model_name = 'roberta-base'
	# elif model == 'BIOMED_ROBERTA':
		# model_name = 'allenai/biomed_roberta_base'
		# assert not is_tf, 'Not TF model found for ' + model_name
	
	model_name = is_tf and model_dict['tf'][model] or model_dict['pytorch'][model]
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	if is_tf and (model=='BERT' or model=='DEBERT'):
		Model = TFBertForNextSentencePrediction.from_pretrained(model_name)
	elif is_tf:
		Model = TFAutoModel.from_pretrained(model_name)
	elif not is_tf and model=='BIOCLINICALBERT':
		Model = BertForNextSentencePrediction.from_pretrained(model_name)
	else:
		Model = AutoModel.from_pretrained(model_name, return_dict=True)
		

	labels = set()
	sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

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

			sentences = []
			for si in sections:
				st = sent_tokenize(si['text'])
				sentences.extend(st)
			
			sentences = sentences[:max_sent]

			for i in range(len(sentences) - 1):
			# create pairwise sentence embeddings
				if is_tf:
					encoding = tokenizer(sentences[i],sentences[i+1], return_tensors="tf",padding=True,truncation=True,max_length=512)
					if model == 'BERT' or model == 'DEBERT':
						# for BERT and DEBERT with Tensorflow
						# we have extracted CLS and SEP embeddings
						output = Model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'], output_hidden_states=True, return_dict=True)
						sep_idx = np.where(encoding['input_ids'][0]==sep_id)[0][0]
						sents.append({'cls': str(output['hidden_states'][-1][:,0,:].numpy().tolist()[0]), 'sep': str(output['hidden_states'][-1][:,sep_idx,:].numpy().tolist()[0])})
					elif model == 'XLNET':
						# for XLNET with Tensorflow
						output = Model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'], return_dict=True)
						sep_idx = np.where(encoding['input_ids'][0]==sep_id)[0][0]
						sents.append({'cls': str(output['last_hidden_state'][:,-1,:].numpy().tolist()[0]), 'sep': str(output['last_hidden_state'][:,sep_idx,:].numpy().tolist()[0])})
					elif model == 'ALBERT':
						# for ALBERT with Tensorflow
						output = Model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'], return_dict=True)
						sep_idx = np.where(encoding['input_ids'][0]==sep_id)[0][0]
						sents.append({'cls': str(output['last_hidden_state'][:,0,:].numpy().tolist()[0]), 'sep': str(output['last_hidden_state'][:,sep_idx,:].numpy().tolist()[0])})
					elif model == 'ROBERTA':
						# for ROBERTA with Tensorflow
						output = Model(encoding['input_ids'], return_dict=True)
						sep_idx = np.where(encoding['input_ids'][0]==sep_id)[0][0]
						sents.append({'cls': str(output['last_hidden_state'][:,0,:].numpy().tolist()[0]), 'sep': str(output['last_hidden_state'][:,sep_idx,:].numpy().tolist()[0])})
				elif not is_tf and model == 'BIOCLINICALBERT':
					# for BIOCLINICALBERT with Pytorch
					encoding = tokenizer(sentences[i],sentences[i+1], return_tensors="pt",padding=True,truncation=True,max_length=512)
					sep_idx = np.where(encoding['input_ids'][0]==sep_id)[0][0]
					if torch.cuda.is_available():  
						dev = "cuda:0" 
						inputs = assign_GPU_pt(encoding)
						Model = Model.to(dev)
						output = Model(**inputs, output_hidden_states = True, return_dict=True)
						sents.append({'cls': str(output['hidden_states'][:,0,:].cpu().detach().numpy().tolist()[0]), 'sep': str(output['hidden_states'][:,sep_idx,:].cpu().detach().numpy().tolist()[0])})
					else:  
						output = Model(**encoding, output_hidden_states = True, return_dict=True)
						sents.append({'cls': str(output['hidden_states'][:,0,:].detach().numpy().tolist()[0]), 'sep': str(output['hidden_states'][:,sep_idx,:].detach().numpy().tolist()[0])})
				elif not is_tf and model == 'BIOMED_ROBERTA':
					# for BIOMED_ROBERTA with Pytorch
					encoding = tokenizer(sentences[i],sentences[i+1], return_tensors="pt",padding=True,truncation=True,max_length=512)
					sep_idx = np.where(encoding['input_ids'][0]==sep_id)[0][0]
					if torch.cuda.is_available():  
						dev = "cuda:0" 
						inputs = assign_GPU_pt(encoding)
						Model = Model.to(dev)
						output = Model(**inputs)
						sents.append({'cls': str(output.last_hidden_state[:,0,:].cpu().detach().numpy().tolist()[0]), 'sep': str(output.last_hidden_state[:,sep_idx,:].cpu().detach().numpy().tolist()[0])})
					else:  
						output = Model(**encoding, output_hidden_states = True, return_dict=True)
						sents.append({'cls': str(output.last_hidden_state[:,0,:].detach().numpy().tolist()[0]), 'sep': str(output.last_hidden_state[:,sep_idx,:].detach().numpy().tolist()[0])})
			processed_data.append(sents)

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
	
	assert args.tf == 'T' or args.tf == 'F', 'tf can be T or F'
	if args.tf == 'T':
		is_tf = True
	elif args.tf == 'F':
		is_tf = False
	
		
	prep_data(args.model,args.corpus,max_sent,test_d_size,is_tf,args.train=='T',args.validation=='T',args.test=='T')
	
