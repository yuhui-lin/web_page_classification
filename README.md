This repository contains Deep learning algorithms for web page classification written in Tensorflow (Python).

## Requirements
```
mkvirtualenv --python=`which python3` --system-site-packages wpc
pip install -r requirements.txt
```
- Python 3.2+
- numpy
- TensorFlow 0.8+
- [NLTK 3.0](http://www.nltk.org/install.html)
- [google 1.9.1+](https://pypi.python.org/pypi/google)

## Running
```
python -m collect --data_dir ~/Downloads/wpc --dataset_type dmoz --max_file_num 5
python -m collect --data_dir ~/Downloads/wpc --pages_per_file 100 --max_file_num 5 --dataset_type dmoz --cat_num 5
python -m collect --data_dir ~/Downloads/wpc --pages_per_file 2000 --max_file_num 5 --dataset_type dmoz --cat_num 5
python -m collect --data_dir ~/Downloads/wpc --pages_per_file 1000 --max_file_num 5 --dataset_type dmoz --cat_num 10
python -m collect --data_dir ~/Downloads/wpc --pages_per_file 100 --max_file_num 5 --dataset_type ukwa

python -m convert --data_dir /media/yuhui/linux/wpc_data/ --dataset_type dmoz --num_cats 5 --html_folder html_5
python -m convert --data_dir ~/Downloads/wpc --dataset_type dmoz --num_cats 5 --html_folder html_5-2500
python -m convert --data_dir ~/Downloads/wpc --dataset_type dmoz --num_cats 10 --html_folder html_10
python -m convert --dataset_type dmoz --num_cats 10 --html_folder html_10
python -m convert --data_dir ~/Downloads/wpc --dataset_type dmoz --num_cats 5 --html_folder html_test --verbose False

python -m train --data_dir ~/Downloads/wpc/ --dataset dmoz-10 --model_type resnn --print_step 3 --summary_step 40 --checkpoint_step 500
python -m train --data_dir /media/yuhui/linux/wpc_data/ --dataset dmoz-10 --model_type resnn  --print_step 3 --summary_step 50 --checkpoint_step 300
python -m train --data_dir ~/Downloads/wpc/ --dataset dmoz-5-2500 --model_type resnn --if_eval False 
python -m train --data_dir ~/Downloads/wpc/ --dataset dmoz-10 --model_type resnn --print_step 2 --summary_step 50 -- checkpoint_step 1000

python -m train --data_dir ~/Downloads/wpc/ --num_cats 5 --model_type cnn --tfr_folder TFR_5-2500
python -m train --data_dir ~/Downloads/wpc/ --num_cats 5 --model_type cnn --tfr_folder TFR_5-2500 --if_eval True

python -m eval --data_dir ~/Downloads/wpc/ --model_type cnn --train_dir ~/Downloads/wpc/cnn/outputs/
```


## Modules

convert: to shuffle all category samples thoroughly, create only two files first: examples.train and examples.test. These two will be split into several TFRecord files which might more than 1 GB. A new argument: the max samples per TFR

## License
MIT

##Note
### googlescraper
put chromedriver under ~/work/py-envs/wpc/bin/
