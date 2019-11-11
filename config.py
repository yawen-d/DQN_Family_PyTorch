class AgentConfig:
    EXPERIMENT_NO = 21

    START_EPISODE = 0
    NUM_EPISODES = 1000
    MEMORY_CAPA = 100000
    MAX_EPS = 1.0
    MIN_EPS = 0.05
    
    LR = 5e-4
    DECAY_RATE = 0.99
    BATCH_SIZE = 32
    UPDATE_FREQ = 10
    GAMMA = 0.99

    
    RES_PATH = './experiments/'

    DOUBLE = False
    DUELING = True
    PER = False


class EnvConfig:
    ENV = "CartPole-v0"