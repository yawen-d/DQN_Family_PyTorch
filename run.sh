# python main.py -x 10 -lr 5e-4
# python main.py -x 11 -lr 25e-5

# python main.py -x 20 -lr 5e-4 --double
# python main.py -x 21 -lr 25e-5 --double

# python main.py -x 30 -lr 5e-4 --dueling
# python main.py -x 31 -lr 25e-5 --dueling

# python main.py -x 40 -lr 5e-4 --per 
# python main.py -x 41 -lr 25e-5 --per

# python main.py -x 50 -lr 5e-4 --dueling --per
# python main.py -x 51 -lr 25e-5 --dueling --per

# python main.py -x 60 -lr 5e-4 --double --per
# python main.py -x 61 -lr 25e-5 --double --per

# python main.py -x 70 -lr 5e-4 --double --dueling 
# python main.py -x 71 -lr 25e-5 --double --dueling 

# python main.py -x 80 -lr 5e-4 --double --dueling --per
# python main.py -x 81 -lr 25e-5 --double --dueling --per

python main.py -x 90 -lr 5e-4 --double --dueling --per --env "CartPole-v1"
# python main.py -x 91 -lr 25e-5 --double --dueling --per --env "CartPole-v1"