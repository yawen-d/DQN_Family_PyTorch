### 1x-8x DQN variants 
### x=0,1: greedy action based on target net + (*abs_errors & glNorm_ISW)?
### x=2: greedy action based on policy net + glNorm_ISW - *abs_errors

# python main.py -x 10 -lr 5e-4
# python main.py -x 11 -lr 25e-5
python main.py -x 12 -lr 5e-4 \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 20 -lr 5e-4 --double
# python main.py -x 21 -lr 25e-5 --double
python main.py -x 22 -lr 5e-4 --double \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 30 -lr 5e-4 --dueling
# python main.py -x 31 -lr 25e-5 --dueling
python main.py -x 32 -lr 5e-4 --dueling \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 40 -lr 5e-4 --per 
# python main.py -x 41 -lr 25e-5 --per
python main.py -x 42 -lr 5e-4 --per \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 50 -lr 5e-4 --dueling --per
# python main.py -x 51 -lr 25e-5 --dueling --per
python main.py -x 52 -lr 5e-4 --dueling --per \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 60 -lr 5e-4 --double --per
# python main.py -x 61 -lr 25e-5 --double --per
python main.py -x 62 -lr 5e-4 --double --per \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 70 -lr 5e-4 --double --dueling 
# python main.py -x 71 -lr 25e-5 --double --dueling 
python main.py -x 72 -lr 5e-4 --double --dueling \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

# python main.py -x 80 -lr 5e-4 --double --dueling --per
# python main.py -x 81 -lr 25e-5 --double --dueling --per
python main.py -x 82 -lr 5e-4 --double --dueling --per \
--num_episodes 1000 --lr_step_size 200 --env "CartPole-v0" 

### looking for the king policy

## No.90 norm_ISW + *abs_errors + lr_step_size: 200 + target net greedy action 
# python main.py -x 90 -lr 5e-4 --double --dueling --per --env "CartPole-v1"

## No.91 glNorm_ISW + *abs_errors + lr_step_size: 400 + target net greedy action 
# python main.py -x 91 -lr 5e-4 --double --dueling --per --env "CartPole-v1"

## No.92 glNorm_ISW + *abs_errors + lr_step_size: 300 + policy net greedy action 
# python main.py -x 92 -lr 5e-4 --double --dueling --per --env "CartPole-v1" \
# --gamma 0.99 --lr_step_size 300 --num_episodes 1500

## No.93 glNorm_ISW + *abs_errors + lr_step_size: 400 + policy net greedy action 
# python main.py -x 93 -lr 5e-4 --double --dueling --per --env "CartPole-v1" --gamma 0.995