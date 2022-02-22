export CUDA_VISIBLE_DEVICES=0

prepare-dataset:
	mkdir -p iemocap
	python preprocess.py ${fold} iemocap/ false

extract-features:
	mkdir -p time-${feat}
	python data.py iemocap/train-${fold}.csv tokenizer-${fold}.pkl time-${feat}/train-${fold}.npz ${feat} false
	python data.py iemocap/test-${fold}.csv tokenizer-${fold}.pkl time-${feat}/test-${fold}.npz ${feat} false

run-experiment:
	mkdir -p models
	python main.py -mode train -source ${source} -fusion ${fusion} \
		-tokenizer_filename tokenizer-${fold}.pkl -train_filename time-${feat}/train-${fold}.npz
	python main.py -mode test -source ${source} -fusion ${fusion} \
		-tokenizer_filename tokenizer-${fold}.pkl -test_filename time-${feat}/test-${fold}.npz
