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
python -m collect --data_dir ~/Downloads/wpc --pages_per_file 10 --max_file_num 5 --dataset_type dmoz --cat_num 5
python -m collect --data_dir ~/Downloads/wpc --pages_per_file 3 --max_file_num 2 --dataset_type dmoz

python -m convert --data_dir ~/Downloads/wpc --html_folder html_

python -m cnn.train --data_dir ~/Downloads/wpc/
python -m cnn.eval --data_dir ~/Downloads/wpc/ --train_dir ~/Downloads/wpc/cnn/outputs/
```


## Modules

## License
MIT

##Note
### googlescraper
put chromedriver under ~/work/py-envs/wpc/bin/
