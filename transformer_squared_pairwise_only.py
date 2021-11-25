import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import activations
import random
from itertools import product, combinations
import math
import json
import argparse
from sklearn.utils import shuffle
import glob
import os
import segeval
from tqdm import tqdm

TRAIN_DATA = []
DEV_DATA = []
TEST_DATA = []
PAD = '[pad]'

#------------------------model-------------------
def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates
	
def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
						  np.arange(d_model)[np.newaxis, :],
						  d_model)
	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	
	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	
	pos_encoding = angle_rads[np.newaxis, ...]
	
	return tf.cast(pos_encoding, dtype=tf.float32)
	
def scaled_dot_product_attention(q, k, v, mask):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead) 
	but it must be broadcastable for addition.
  
	Args:
		q: query shape == (..., seq_len_q, depth)
		k: key shape == (..., seq_len_k, depth)
		v: value shape == (..., seq_len_v, depth_v)
		mask: Float tensor with shape broadcastable 
			to (..., seq_len_q, seq_len_k). Defaults to None.
		
	Returns:
		output, attention_weights
	"""

	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
	
	# scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)  

	# softmax is normalized on the last axis (seq_len_k) so that the scores
	# add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

	output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

	return output, attention_weights
	
class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		
		assert d_model % self.num_heads == 0
		
		self.depth = d_model // self.num_heads
		
		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)
		
		self.dense = tf.keras.layers.Dense(d_model)
			
	def split_heads(self, x, batch_size):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])
		
	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]
		
		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)
		
		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
		
		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)
		
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention, 
									(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
			
		return output, attention_weights
		
def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
		tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
	])
	
class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer, self).__init__()

		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		
	def call(self, x, training, mask):

		attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
		
		ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
		
		return out2
		
class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff,
				maximum_position_encoding, rate=0.1):
		super(Encoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers
		
		self.embedding = tf.keras.layers.InputLayer(input_shape=(maximum_position_encoding,self.d_model))
		self.pos_encoding = positional_encoding(maximum_position_encoding, 
												self.d_model)
		
		
		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
						for _ in range(num_layers)]
	
		self.dropout = tf.keras.layers.Dropout(rate)
			
	def call(self, x, training, mask):

		seq_len = tf.shape(x)[1]
		
		# adding embedding and position encoding.
		x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_encoding[:, :seq_len, :]

		x = self.dropout(x, training=training)
		
		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)
		
		return x  # (batch_size, input_seq_len, d_model)
		

class Joint_Tobert_TextSeg(tf.keras.Model):
	def __init__(self, seq_len=150, embed_dim=768, begin_output_dim=4, label_output_dim=4, drop_rate=0.1, attn_head=8, encoder_layers=2, encoder_dff=30):
		super(Joint_Tobert_TextSeg, self).__init__()
		self.encoder_block = Encoder(num_layers=encoder_layers, d_model=embed_dim, num_heads=attn_head, 
						 dff=encoder_dff,
						 maximum_position_encoding=seq_len)
		self.masking1 = tf.keras.layers.Masking()
		self.masking2 = tf.keras.layers.Masking()
		self.dropout1 = tf.keras.layers.Dropout(drop_rate)
		self.dense1 = tf.keras.layers.Dense(begin_output_dim, activation="softmax", name="begin_output")
		self.dense2 = tf.keras.layers.Dense(label_output_dim, activation="softmax", name="label_output")
		
	def call(self, inputs, training=False):
		x = self.encoder_block(inputs[0], training=training, mask=inputs[1])
		x1 = self.masking1(tf.math.multiply(x, inputs[2]))
		x1 = self.dropout1(x1,training=training)
		x2 = self.masking2(tf.math.multiply(x, inputs[3]))
		x2 = self.dropout1(x2,training=training)
		return [self.dense1(x1),self.dense2(x2)]


#--------------------data--------------------------

def get_begin_dict():
	begin_dict = {
		PAD: 0,
		0: 1,
		1: 2
	}
	return begin_dict
	
def get_label_dict(model = 'BERT', corpus='en_city'):
	label_dict = {}
	with open(os.path.join('processed_data', 'wikisection_' + corpus + '_label_dict'), 'r+') as f:
		for l in f.readlines():
			key, value = l.strip().split()
			label_dict[key] = int(value)
	return label_dict

def load_train_data(model = 'BERT', corpus='en_city'):
	print('loading train data',end='...')
	with open(os.path.join('processed_data', model, 'wikisection_' + corpus + '_train.json'), 'r+') as f:
		original = json.load(f)
	with open(os.path.join('processed_data', model, 'wikisection_' + corpus + '_pairwise_train.json'), 'r+') as f:
		pairwise = json.load(f)

	assert len(original) == len(pairwise), 'data size not match'
	train_data = []
	for idx in tqdm(range(len(original))):
		data = original[idx]
		data['pairwise'] = pairwise[idx]
		train_data.append(data)
	TRAIN_DATA.extend(train_data)
	
	print('Done!')

	
def load_dev_data(model = 'BERT', corpus='en_city'):
	print('loading dev data',end='...')
	with open(os.path.join('processed_data', model, 'wikisection_' + corpus + '_validation.json'), 'r+') as f:
		original = json.load(f)
	with open(os.path.join('processed_data', model, 'wikisection_' + corpus + '_pairwise_validation.json'), 'r+') as f:
		pairwise = json.load(f)

	assert len(original) == len(pairwise), 'data size not match'
	dev_data = []
	for idx in tqdm(range(len(original))):
		data = original[idx]
		data['pairwise'] = pairwise[idx]
		dev_data.append(data)
	
	DEV_DATA.extend(dev_data)
	print('Done!')
	
def load_test_data(model = 'BERT', corpus='en_city'):
	print('loading test data',end='...')
	with open(os.path.join('processed_data', model, 'wikisection_' + corpus + '_test.json'), 'r+') as f:
		original = json.load(f)
	with open(os.path.join('processed_data', model, 'wikisection_' + corpus + '_pairwise_test.json'), 'r+') as f:
		pairwise = json.load(f)

	assert len(original) == len(pairwise), 'data size not match'
	test_data = []
	for idx in tqdm(range(len(original))):
		data = original[idx]
		data['pairwise'] = pairwise[idx]
		test_data.append(data)
	
	TEST_DATA.extend(test_data)
	print('Done!')

def get_test_true_begin(seq_len = 150, flattened = True):
	batch_output = []
	begin_dict = get_begin_dict()
	if flattened:
		for doc in TEST_DATA:
			for sent in doc['sent'][:seq_len]:
				batch_output.append(begin_dict[sent['begin']])
			for i in range(max(seq_len-len(doc['sent']), 0)):
				batch_output.append(begin_dict[PAD])
	else:
		for doc in TEST_DATA:
			doc_output = []
			for sent in doc['sent'][:seq_len]:
				doc_output.append(begin_dict[sent['begin']])
			for i in range(max(seq_len-len(doc['sent']), 0)):
				doc_output.append(begin_dict[PAD])
			batch_output.append(doc_output)
	return batch_output
	
def get_test_true_label(model='BERT', seq_len = 150, flattened = True, corpus = 'en_city'):
	batch_output = []
	label_dict = get_label_dict(model, corpus)
	if flattened:
		for doc in TEST_DATA:
			for sent in doc['sent'][:seq_len]:
				batch_output.append(label_dict[sent['label']])
			for i in range(max(seq_len-len(doc['sent']), 0)):
				batch_output.append(label_dict[PAD])
	else:
		for doc in TEST_DATA:
			doc_output = []
			for sent in doc['sent'][:seq_len]:
				doc_output.append(label_dict[sent['label']])
			for i in range(max(seq_len-len(doc['sent']), 0)):
				doc_output.append(label_dict[PAD])
			batch_output.append(doc_output)
	return batch_output

def gen_batch_input(data, model = 'BERT', seq_len=72, embed_dim=768, mask_b_rate = 1, mask_i_rate = 0.5, corpus = 'en_city'):
	"""
	:param	mask_b_rate: probability of begin segment remains
	:param	mask_i_rate: probability of inner segment remains
	"""
	begin_dict = get_begin_dict()
	label_dict = get_label_dict(model, corpus)
	batch_input_cls = []
	batch_input_eval_mask = []
	batch_input_transformer_mask = []
	batch_segment_output = []
	batch_label_output = []
	for doc in data:
		doc_chunk_len = len(doc['sent'])
		input_cls = [[0.0001] * embed_dim] + [eval(sent['cls']) for sent in doc['pairwise']] + [[0.] * embed_dim] * max(seq_len - doc_chunk_len, 0)
		input_cls = input_cls[:seq_len]
		input_eval_mask = []
		input_transformer_mask = []
		for sent in doc['sent']:
			r = np.random.random()
			if (sent['begin'] == 0 and r > mask_i_rate) or (sent['begin'] == 1 and r > mask_b_rate):
				# replace output to 0
				input_eval_mask.append(1.)
			else:
				input_eval_mask.append(0.)
			input_transformer_mask.append(0.)
		input_eval_mask = input_eval_mask + [1.] * max(seq_len - doc_chunk_len, 0)
		input_eval_mask = input_eval_mask[:seq_len]
		input_transformer_mask = input_transformer_mask + [1.] * max(seq_len - doc_chunk_len, 0)
		input_transformer_mask = input_transformer_mask[:seq_len]
		segment_output = [ [begin_dict[sent['begin']]] for sent in doc['sent'] ]
		segment_output = segment_output + [[begin_dict[PAD]]] * max(seq_len - doc_chunk_len, 0)
		segment_output = segment_output[:seq_len]
		label_output = [ [label_dict[sent['label']]] for sent in doc['sent'] ]
		label_output = label_output + [[label_dict[PAD]]] * max(seq_len - doc_chunk_len, 0)
		label_output = label_output[:seq_len]
		batch_input_cls.append(input_cls)
		batch_input_eval_mask.append(input_eval_mask)
		batch_input_transformer_mask.append(input_transformer_mask)
		batch_segment_output.append(segment_output)
		batch_label_output.append(label_output)
		
	return [np.asarray(batch_input_cls),np.asarray(batch_input_transformer_mask)[:,np.newaxis,np.newaxis,:],np.not_equal(np.expand_dims(batch_input_eval_mask,axis=-1),1).astype(float),np.not_equal(np.expand_dims(batch_input_transformer_mask,axis=-1),1).astype(float)], [np.asarray(batch_segment_output),np.asarray(batch_label_output)]
	
def train_generator(batch_size, model = 'BERT', seq_len=72, embed_dim=768, mask_b_rate = 1, mask_i_rate = 0.5, corpus = 'en_city'):
	_train_data = TRAIN_DATA
	while True:
		_train_data = shuffle(_train_data)
		for i in range(0,len(_train_data),batch_size):
			yield gen_batch_input(_train_data[i:i+batch_size], model=model, seq_len=seq_len,embed_dim=embed_dim, mask_b_rate = mask_b_rate, mask_i_rate = mask_i_rate, corpus = corpus)
		
def dev_generator(batch_size, model = 'BERT', seq_len=72, embed_dim=768, mask_b_rate = 1, mask_i_rate = 0.5, corpus = 'en_city'):
	while True:
		for i in range(0,len(DEV_DATA),batch_size):
			yield gen_batch_input(DEV_DATA[i:i+batch_size], model=model,seq_len=seq_len,embed_dim=embed_dim, mask_b_rate = mask_b_rate, mask_i_rate = mask_i_rate, corpus = corpus)
			
def test_generator(batch_size, model = 'BERT', seq_len=72, embed_dim=768, corpus = 'en_city'):
	while True:
		for i in range(0,len(TEST_DATA),batch_size):
			#taking every sentence into account
			yield gen_batch_input(TEST_DATA[i:i+batch_size], model=model,seq_len=seq_len,embed_dim=embed_dim, mask_b_rate = 1, mask_i_rate = 1, corpus = corpus)
		


	
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', help="BERT/BLUEBERT", default='en_city')
	parser.add_argument('--corpus', help="corpus:en_city/en_disease/de_city/de_disease", default='en_city')
	parser.add_argument('--seq_len', help="maximum number of document chunks", default='150')
	parser.add_argument('--embed_dim', help="input chunk cls dimension", default='768')
	parser.add_argument('--attn_head', help="number of attention heads", default='12')
	parser.add_argument('--encoder_layers', help="number of encoder layers", default='10')
	parser.add_argument('--encoder_dff', help="dimension of feed forward pointwise network", default='768')
	parser.add_argument('--epoch', help="number of epochs in training", default='100')
	parser.add_argument('--batch_size', help="batch size", default='32')
	parser.add_argument('--mask_b_rate', help="probability of begin segment remains", default='1')
	parser.add_argument('--mask_i_rate', help="probability of inner segment remains", default='0.5')
	parser.add_argument('--train_step', help="number of training steps for each epoch", default='100')
	parser.add_argument('--val_step', help="number of validation steps for each epoch", default='100')
	parser.add_argument('--lr', help="learning rate", default='0.0001')
	parser.add_argument('--patience', help="learning rate", default='10')
	parser.add_argument('--output_model', help="output model name", default='model_test')
	parser.add_argument('--output_result', help="output model name prefix", default='result_test')
	
	args = parser.parse_args()
	
	assert args.corpus in ['en_city','en_disease','de_city','de_disease'], "Invalid corpus"
	assert args.seq_len.isdigit(),'seq_len must be integer'
	assert args.embed_dim.isdigit(),'embed_dim must be integer'
	assert args.attn_head.isdigit(),'attn_head must be integer'
	assert args.epoch.isdigit(),'epoch must be integer'
	assert args.batch_size.isdigit(),'batch_size must be integer'
	assert args.train_step.isdigit(),'train_step must be integer'
	assert args.val_step.isdigit(),'val_step must be integer'
	assert args.encoder_layers.isdigit(),'encoder_layers must be integer'
	assert args.encoder_dff.isdigit(),'encoder_dff must be integer'
	assert args.patience.isdigit(),'patience must be integer'
	try:
		float(args.lr)
		float(args.mask_b_rate)
		float(args.mask_i_rate)
	except ValueError:
		raise "lr/mask_b_rate/mask_i_rate must be float"
	
	print(args)
	
	load_train_data(model = args.model, corpus=args.corpus)
	load_dev_data(model = args.model, corpus=args.corpus)
	optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.lr))
	
	model = Joint_Tobert_TextSeg(seq_len=int(args.seq_len), embed_dim=int(args.embed_dim), begin_output_dim=len(get_begin_dict()), label_output_dim=len(get_label_dict(model = args.model, corpus=args.corpus)), attn_head=int(args.attn_head), encoder_layers = int(args.encoder_layers),encoder_dff = int(args.encoder_dff))
	
	loss = tf.keras.losses.SparseCategoricalCrossentropy()
	print('Compiling model')
	model.compile(loss=loss, optimizer=optimizer,metrics = tf.keras.metrics.SparseCategoricalAccuracy())
	print('Training model')
	model.fit(train_generator(batch_size = int(args.batch_size), model=args.model, seq_len=int(args.seq_len), embed_dim=int(args.embed_dim), mask_b_rate = float(args.mask_b_rate), mask_i_rate = float(args.mask_i_rate), corpus = args.corpus),
				steps_per_epoch=int(args.train_step),
				epochs=int(args.epoch),
				validation_data=dev_generator(batch_size = int(args.batch_size), model=args.model, seq_len=int(args.seq_len), embed_dim=int(args.embed_dim), mask_b_rate = float(args.mask_b_rate), mask_i_rate = float(args.mask_i_rate), corpus = args.corpus),
				validation_steps=int(args.val_step),
				callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(args.patience))
				],)
	# print('Saving model')
	# model.save(args.output_model)
	
	print('Evaluating model')
	
	del train_generator
	del dev_generator
	del TRAIN_DATA
	del DEV_DATA
	load_test_data(model = args.model, corpus=args.corpus)
	true_begin = get_test_true_begin(seq_len=int(args.seq_len))
	true_label = get_test_true_label(model=args.model, seq_len=int(args.seq_len), corpus = args.corpus)
	predictions = model.predict(test_generator(batch_size = int(args.batch_size), model=args.model, seq_len=int(args.seq_len), embed_dim=int(args.embed_dim), corpus = args.corpus),steps=math.ceil(len(TEST_DATA)/int(args.batch_size)))
	begin_predictions_prob = predictions[0]
	begin_predictions = np.argmax(begin_predictions_prob,axis=-1)
	label_predictions = predictions[1]
	#print(begin_predictions.shape)
	label_predictions
	#pred_prob_flattened = []
	
	# for j in predictions.tolist():
		# for i in j:
			# pred_prob_flattened.append(i)
	begin_pred_flattened = []
	for i in begin_predictions:
		begin_pred_flattened.extend(i)
	
	label_pred_flattened = []
	for i in label_predictions:
		label_pred_flattened.extend(np.argmax(i,axis=-1))
	# pred = np.argmax(np.asarray(predictions),axis= -1)
	
	true_pred_begin_pairs = list(zip(get_test_true_begin(seq_len=int(args.seq_len),flattened=False),begin_predictions))
	
	pk_scores = []
	for pair in true_pred_begin_pairs:
		# document
		true_seg = []
		pred_seg = []
		true_count = 0
		pred_count = 0
		for j in range(len(pair[0])):
			if pair[0][j] == 0:
				break
			if pair[0][j] == 2:
				if true_count != 0:
					true_seg.append(true_count)
				true_count = 1
			else:
				true_count += 1
			if pair[1][j] == 2:
				if pred_count != 0:
					pred_seg.append(pred_count)
				pred_count = 1
			else:
				pred_count += 1
		true_seg.append(true_count)
		pred_seg.append(pred_count)
		pk_scores.append(segeval.pk(pred_seg,true_seg,window_size=10))
	
	# print('Outputting result')
	# with open('results/' + 'tobert_masked_crf_' + args.output_result + '_prob', 'w+') as f:
		# for item in list(zip(true_targets,pred_prob_flattened)):
			# f.write(str(item[0]) + ',' + str(item[1]) + '\n')
	# with open('results/'+ 'tobert_masked_joint_' + args.output_result + '_begin_record', 'w+') as f:
		# for item in list(zip(true_begin,begin_pred_flattened)):
			# f.write(str(item[0]) + ',' + str(item[1]) + '\n')
			
	# with open('results/'+ 'tobert_masked_joint_' + args.output_result + '_label_record', 'w+') as f:
		# for item in list(zip(true_label,label_pred_flattened)):
			# f.write(str(item[0]) + ',' + str(item[1]) + '\n')
	
	print('outputing metrics')
	print('pk: ' + str(sum(pk_scores)/len(pk_scores)))
	with open('results/' + 'tobert_masked_joint_' + args.output_result + '_metrics.txt', 'w+') as f:
		# eval_metrics = model.evaluate(test_generator(batch_size = int(args.batch_size), seq_len=int(args.seq_len), embed_dim=int(args.embed_dim)),steps=math.ceil(len(TEST_DATA)/int(args.batch_size)))
		# eval_metrics = list(zip(model.metrics_names,eval_metrics))
		f.write(str(args) + '\n')
		f.write('pk: ' + str(sum(pk_scores)/len(pk_scores)) + '\n')
		# for i in eval_metrics:
			# f.write(str(i[0]) + ': ' + str(i[1]) + '\n')
			