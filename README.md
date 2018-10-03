# POS Tagging for Chinese

Model : CharLSTM + LSTM + CRF

Pretrained Word Embedding : giga.100.txt

Data : CTB5

## requirements

```
python >= 3.6.3
pytorch = 0.4.1
```

## running

```
python train.py --pre_emb	# train the model with pretrained embedding
python evaluate 			# reload the model and evaluate test file
```

## results

```
dev = 96.00%
test = 95.68%
```

