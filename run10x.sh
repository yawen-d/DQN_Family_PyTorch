# 101-109 normal DQN with normal greedy action 
# based on policy net
python main.py -x 101 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 32 --gamma 0.99 

python main.py -x 102 --num_episodes 1000 --env "CartPole-v0" \
-lr 1e-4 --decay_rate 0.99 --batch_size 32 --gamma 0.99 
python main.py -x 103 --num_episodes 1000 --env "CartPole-v0" \
-lr 1e-3 --decay_rate 0.99 --batch_size 32 --gamma 0.99 
python main.py -x 104 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 32 --gamma 0.99 --lr_step_size 200

python main.py -x 105 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 64 --gamma 0.99 
python main.py -x 106 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 16 --gamma 0.99 

python main.py -x 107 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 32 --gamma 0.999
python main.py -x 108 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 32 --gamma 0.995
python main.py -x 109 --num_episodes 1000 --env "CartPole-v0" \
-lr 5e-4 --decay_rate 0.99 --batch_size 32 --gamma 0.90
 