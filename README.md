# Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence
This is the implementation of Transformer<sup>2</sup> from the following paper:<br />
Kelvin Lo, Yuan Jin, Weicong Tan, Ming Liu, Lan Du, Wray Buntine. <a href='https://aclanthology.org/2021.findings-emnlp.283/' target='_blank'>Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence</a>
```
@inproceedings{lo-etal-2021-transformer-pre,
    title = "Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence",
    author = "Lo, Kelvin  and
      Jin, Yuan  and
      Tan, Weicong  and
      Liu, Ming  and
      Du, Lan  and
      Buntine, Wray",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.283",
    pages = "3334--3340",
    abstract = "This paper proposes a transformer over transformer framework, called Transformer{\^{}}2, to perform neural text segmentation. It consists of two components: bottom-level sentence encoders using pre-trained transformers, and an upper-level transformer-based segmentation model based on the sentence embeddings. The bottom-level component transfers the pre-trained knowledge learnt from large external corpora under both single and pair-wise supervised NLP tasks to model the sentence embeddings for the documents. Given the sentence embeddings, the upper-level transformer is trained to recover the segmentation boundaries as well as the topic labels of each sentence. Equipped with a multi-task loss and the pre-trained knowledge, Transformer{\^{}}2 can better capture the semantic coherence within the same segments. Our experiments show that (1) Transformer{\^{}}2{\$}manages to surpass state-of-the-art text segmentation models in terms of a commonly-used semantic coherence measure; (2) in most cases, both single and pair-wise pre-trained knowledge contribute to the model performance; (3) bottom-level sentence encoders pre-trained on specific languages yield better performance than those pre-trained on specific domains.",
}
```

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
	<tr><td>&lt;model_name&gt;</td><td>ALBERT/BERT/BERTL/BLUEBERT/BIOCLINICALBERT/BIOMED_ROBERTA/DEBERT/XLNET</td></tr>
	<tr><td>&lt;corpus_name&gt;</td><td>en_city/en_disease/de_city/de_disease</td></tr>
	<tr><td>&lt;max_sent&gt;</td><td>Integer</td></tr>
	<tr><td>&lt;is_tensorflow&gt;</td><td>T/F</td></tr>
</table>

### 1. Prepare topic labels lookup

```
python create_label_dict.py

--corpus <corpus_name> : the corpus build topic labels lookup
```

### 2. Prepare single sentence embeddings
To prepare sentence embeddings of single sentence

```
python prepare_single_sentence_embeddings.py

--model <model_name> : pre-trained transformer used to encode sentence
--corpus <corpus_name> : the corpus to encode
--max_sent <max_sent> : maximum number of sentences to encode for each document
--tf <is_tensorflow> : use tensorflow(default: T), use F if <model_name> in ['BIOCLINICALBERT','BIOMED_ROBERTA']
```

### 3. Prepare pairwise sentence embeddings
To prepare sentence embeddings of a pair of consecutive sentences

```
python prepare_pairwise_sentence_embeddings.py

--model <model_name> : pre-trained transformer used to encode sentence
--corpus <corpus_name> : the corpus to encode
--max_sent <max_sent> : maximum number of sentences to encode for each document
--tf <is_tensorflow> : use tensorflow(default: T), use F if <model_name> in ['BIOCLINICALBERT','BIOMED_ROBERTA']
```

### 4. Train and Evaluation

In order to train and test the models, you need to prepare both single and pairwise embeddings.

#### 4.1 Model with single and pairwise sentence embeddings

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

#### 4.2 Model with pairwise sentence embeddings only

```
python transformer_squared_pairwise_only.py

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
python transformer_squared_pairwise_only.py --model BERT --corpus en_disease --attn_head 24 --epoch 500 --batch_size 16 --mask_b_rate 1 --mask_i_rate 0.3 --train_step 100 --val_step 100 --encoder_layers 5 --encoder_dff 1024 --patience 20 --lr 0.0001
```