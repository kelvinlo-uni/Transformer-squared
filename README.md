# Transformer-squared



## Dependencies
This repository is built on Python 3.6.
<ol>
<li>nltk 3.5</li>
<li>numpy 1.18.5</li>
<li>scikit-learn 0.23.2</li>
<li>segeval 2.0.11</li>
<li>tensorflow 2.3.1</li>
<li>torch 1.7</li>
<li>tqdm 4.51</li>
<li>transformers 3.4.0</li>
</ol>

## Usage
<table>
	<thead><td>variable name</td><td>possible values</td></thead>
	<tr><td>&lt;model_name&gt;</td><td>BERT/BERTL/BLUEBERT/BIOCLINICALBERT/DEBERT</td></tr>
	<tr><td>&lt;corpus_name&gt;</td><td>en_city/en_disease/de_city/de_disease</td></tr>
	<tr><td>&lt;max_sent&gt;</td><td>Integer</td></tr>
	<tr><td>&lt;is_tensorflow&gt;</td><td>T/F</td></tr>
</table>

### 1. Prepare single sentence embeddings
To prepare sentence embeddings of single sentence

```
python prepare_single_sentence_embeddings.py

--model <model_name> : pre-trained transformer used to encode sentenve
--corpus <corpus_name> : the corpus to encode
--max_sent <max_sent> : maximum number of sentences to encode for each document
--tf <is_tensorflow> : use tensorflow(default: T), use F if <model_name> in ['BIOCLINICALBERT','BIOMED_ROBERTA']
```

### 2. Prepare pairwise sentence embeddings
To prepare sentence embeddings of a pair of consecutive sentences

```
python prepare_pairwise_sentence_embeddings.py

--model <model_name> : pre-trained transformer used to encode sentenve
--corpus <corpus_name> : the corpus to encode
--max_sent <max_sent> : maximum number of sentences to encode for each document
--tf <is_tensorflow> : use tensorflow(default: T), use F if <model_name> in ['BIOCLINICALBERT','BIOMED_ROBERTA']
```

### 3. Train and Evaluation

#### 3.1 Model with single and pairwise sentence embeddings

```
python transformer_squared_single_pairwise.py

--corpus <corpus_name> : which corpus to train and test on
--seq_len <int> : maximum number of document sentences
--embed_dim <int> : input cls dimension
--attn_head <int> : number of attention heads
--encoder_layers <int> : number of encoder layers
--encoder_dff <int> : dimension of feed forward pointwise network
--epoch <int> : number of epochs in training
--batch_size <int> : batch size
--mask_b_rate <float> : probability of begin segment remains
--mask_i_rate <float> : probability of inner segment remains
--train_step <int> : number of training steps for each epoch
--val_step <int> : number of validation steps for each epoch
--lr <float> : learning rate
--patience <int> : patience of early stopping
--output_result <str>: prefix of output result file
```
To replicate the result in the paper:<br />
```
python transformer_squared_single_pairwise.py --model BERT --corpus en_disease --attn_head 24 --epoch 500 --batch_size 16 --mask_b_rate 1 --mask_i_rate 0.3 --train_step 100 --val_step 100 --encoder_layers 5 --encoder_dff 1024
```