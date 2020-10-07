echo 'run success'
# nohup python driver/Train.py --config_file ./configs/transformer.cfg --thread 1 --gpu 0 >> j2j-news18.log &
# nohup python driver/Train.py --config_file ./configs/fusion-BERT-en.cfg --thread 1 --gpu 2 >> f-j2j-news18.log &
nohup python driver/Train.py --config_file ./configs/DL4MT.cfg --thread 1 --gpu 0 >> lstm-news18.log &