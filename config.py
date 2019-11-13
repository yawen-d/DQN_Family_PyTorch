class EnvConfig:
    ENV = "CartPole-v0"
    
    def get_env_cfg(self, args):
        self.ENV = args['env'] if args['env'] else self.ENV

class AgentConfig:
    EXPERIMENT_NO = 99

    START_EPISODE = 0
    NUM_EPISODES = 1000
    MEMORY_CAPA = 50000
    MAX_EPS = 1.0
    MIN_EPS = 0.01
    UPDATE_FREQ = 10
    DEMO_NUM = 100
    
    LR = 25e-5          # learning rate
    DECAY_RATE = 0.99   # decay rate
    BATCH_SIZE = 32     # batch size
    GAMMA = 0.99        # gamma

    ALPHA = 0.6         # alpha for PER
    BETA = 0.4          # beta for PER

    DOUBLE = False      # double Q-learning
    DUELING = False     # dueling network
    PER = False         # prioritized replay

    RES_PATH = './experiments/'

    def get_agent_cfg(self, args):
        self.EXPERIMENT_NO = args['experiment_num'] if args['experiment_num'] else self.EXPERIMENT_NO
        
        self.LR = args['learning_rate'] if args['learning_rate'] else self.LR
        self.DECAY_RATE = args['decay_rate'] if args['decay_rate'] else self.DECAY_RATE
        self.BATCH_SIZE = args['batch_size'] if args['batch_size'] else self.BATCH_SIZE
        self.GAMMA = args['gamma'] if args['gamma'] else self.GAMMA

        self.ALPHA = args['alpha'] if args['alpha'] else self.ALPHA
        self.BETA = args['beta'] if args['beta']  else self.BETA

        self.DOUBLE = args['double'] if args['double'] else self.DOUBLE
        self.DUELING = args['dueling'] if args['dueling'] else self.DUELING
        self.PER = args['per'] if args['per'] else self.PER