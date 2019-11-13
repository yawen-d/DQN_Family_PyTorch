import argparse

parser = argparse.ArgumentParser(description='Add arguments for DQN configs')
parser.add_argument('-x', '--experiment_num', metavar='N', type=int, \
                    help='experiment number', required=True)
parser.add_argument('-lr', '--learning_rate', metavar='lr', type=float, help='learning rate')
parser.add_argument('-dr', '--decay_rate', metavar='dr', type=float, help='epsilon decay rate')
parser.add_argument('-b', '--batch_size', metavar='bs', type=int, help='batch size')
parser.add_argument('--gamma', metavar='g',  type=float, help='gamma')
parser.add_argument('--alpha', metavar='a',  type=float, help='alpha for PER')
parser.add_argument('--beta', metavar='b',  type=float, help='beta for PER')

parser.add_argument('--double', action='store_true', 
                    help='double DQN')
parser.add_argument('--dueling', action='store_true', 
                    help='dueling DQN')
parser.add_argument('--per', action='store_true', 
                    help='prioritized experience replay')

parser.add_argument('--env', metavar='env', type=str, help='environments')
args = parser.parse_args()

print(vars(args))

