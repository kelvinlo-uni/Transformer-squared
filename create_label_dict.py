import argparse
from pathlib import Path
from tqdm import tqdm
import json


MASK = '[mask]'
PAD = '[pad]'
UNK = '[unk]'
special_tokens = [MASK,PAD,UNK]

corpus = 'en_city' # en_city/en_disease/de_city/de_disease


def prep_data(corpus = 'en_city'):
	original_path = 'data/wikisection/'
	processed_path = 'processed_data/'
	
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
	filenames = ['train.json','validation.json','test.json']

	labels = set()

	for fn in filenames:
		print('Now processing: ' + str(fn))
		with open(original_path + fn, 'r+') as f:
			data = json.load(f)

		processed_data = []
		for data_idx in tqdm(range(0, len(data))):
			data_item = data[data_idx]
			t = data_item['text']
			a = data_item['annotations']
			sections = []
			for ai in a:
				labels.add(ai['sectionLabel'])

	#create label dict
	print('Now creating label dict')
	labels = [PAD,MASK,UNK] + list(labels)
	label_dict = dict([(v, i) for i, v in enumerate(labels)])
	with open(processed_path + 'label_dict', 'w+') as f:
		for key in label_dict:
			f.write(str(key) + ' ' + str(label_dict[key]) + '\n')
	print('Done!')
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--corpus', help="en_city/en_disease/de_city/de_disease")
	
	args = parser.parse_args()
	
		
	prep_data(args.corpus)
	
