
import sys, os, argparse
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), 'class'))

from ParserConf import ParserConf
from DataUtil import DataUtil
import train as starter
import setproctitle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', default='gcncsr', help='model name')
    parser.add_argument('--gpu', nargs='+', help='available gpu id')

    args = parser.parse_args()

    data_name = args.data_name
    model_name = args.model_name
    device_id = ''
    for gid in args.gpu:
        device_id += gid
        device_id += ','
    device_id = device_id.rstrip(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    config_path = os.path.join(os.getcwd(), f'conf/{data_name}/{model_name}.ini')
    conf = ParserConf(config_path)
    conf.parserConf()
    setproctitle.setproctitle('{}_{}_{}'.format(conf.data_name, conf.model_name, conf.test_name))
    data = DataUtil(conf)
    starter.start(conf, data, model_name)
