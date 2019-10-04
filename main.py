from agent import Agent

if __name__ == '__main__':
    agent = Agent()
    agent.train()
    agent.env_close()
    agent.save_results()
