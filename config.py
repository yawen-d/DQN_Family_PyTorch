class AgentConfig:
    EXPERIMENT_NO = 80

    START_EPISODE = 0
    NUM_EPISODES = 500
    MEMORY_CAPA = 10000
    MAX_EPS = 1.0
    MIN_EPS = 0.01
    UPDATE_FREQ = 10
    
    LR = 5e-4          # learning rate
    DECAY_RATE = 0.99   # decay rate
    BATCH_SIZE = 32     # batch size
    GAMMA = 0.99        # gamma

    DOUBLE = True      # double Q-learning
    DUELING = True     # dueling network
    PER = True         # prioritized replay

    RES_PATH = './experiments/'


class EnvConfig:
    ENV = "CartPole-v0"