import argparse
import json

def load_arg(custom_json=None):
    parser = argparse.ArgumentParser(description='FFSNN')
    parser.add_argument('--epoch', type=int, default= 3 , help='local epoch')
    parser.add_argument('--globalepoch', type=int, default= 50 , help='global epoch')
    parser.add_argument('--lr', type=int, default= 0.01 , help='learning rate')
    parser.add_argument('--batchsize', type=int, default= 512 , help='batchsize of the local training')
    parser.add_argument('--algo_type', type=str, default= "supervised" , help='type of SNN')
    parser.add_argument('--seed', type=int, default= 1235 , help= 'seed of random')
    parser.add_argument('--verbosity', type=int, default= 0, help= 'verbosity of the loader')

    parser.add_argument('--nc', type=int, default= 3, help= 'number of the clients')


    # args = parser.parse_args("--lrc 1".split())
    args = parser.parse_args()

    
    return args

if __name__ == "__main__":
    args = load_arg()
    args.nc = 5
    print("{}".format(args.nc))