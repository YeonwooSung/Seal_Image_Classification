import sys


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: main.py [-binary | -multi] [estimator_name]')
        exit(1)

    mode = sys.argv[1]
    estimator_name = sys.argv[2]

    #TODO file path of the data files!!
    data_path = '../data'

    #TODO validate estimator name

    if mode == 'binary':
        print('Start binary classification')
    
        #TODO

    elif mode == 'multi':
        print('Start multi-class classification')
    
        #TODO

    else:
        print('Usage: main.py [-binary | -multi] [estimator_name]')
        exit(1)
