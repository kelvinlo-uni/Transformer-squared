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
</table>

### 1. Prepare single sentence embeddings
To prepare sentence embeddings of single sentence
`python prepare_single_sentence_embeddings.py`
<ul>
<li><pre>--model &lt;model_name&gt;</pre></li>
<li><pre>--corpus &lt;corpus_name&gt;</pre></li>
</ul>