from agent import Agent

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Add arguments for DQN configs')
    parser.add_argument('-x', '--experiment_num', metavar='N', type=int, \
                    help='experiment number', required=True)
    parser.add_argument('-lr', '--learning_rate', metavar='lr', type=float, help='learning rate')
    parser.add_argument('-dr', '--decay_rate', metavar='dr', type=float, help='epsilon decay rate')
    parser.add_argument('-b', '--batch_size', metavar='bs', type=int, help='batch size')
    parser.add_argument('-epi', '--num_episodes', metavar='epi', type=int, help='number of episodes')
    parser.add_argument('--gamma', metavar='g',  type=float, help='gamma')
    parser.add_argument('--alpha', metavar='a',  type=float, help='alpha for PER')
    parser.add_argument('--beta', metavar='b',  type=float, help='beta for PER')
    parser.add_argument('-ss', '--lr_step_size', metavar='ss', type=int, help='learning rate step size')

    parser.add_argument('--double', action='store_true', 
                        help='double DQN')
    parser.add_argument('--dueling', action='store_true', 
                        help='dueling DQN')
    parser.add_argument('--per', action='store_true', 
                        help='prioritized experience replay')

    parser.add_argument('--env', metavar='env', type=str, help='environments')
    return parser.parse_args()


if __name__ == '__main__':
    args = vars(get_args())
    print("------------------")
    for k,v in args.items():
        if v: print("{} = {}".format(k, v))
    print("------------------")
    agent = Agent(args)
    agent.train()
    agent.env_close()
    agent.save_results()