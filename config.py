class AgentConfig:
    EXPERIMENT_NO = 9

    START_EPISODE = 0
    NUM_EPISODES = 1000
    MEMORY_CAPA = 100000
    MAX_EPS = 1.0
    MIN_EPS = 0.05
    
    LR = 5e-4
    DECAY_RATE = 0.999
    BATCH_SIZE = 16
    UPDATE_FREQ = 10
    GAMMA = 0.999

    
    RES_PATH = './experiments/'

    PER = False


class EnvConfig:
    ENV = "CartPole-v0"