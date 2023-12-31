import random
import Vector

MUTATION_RATE = 10

MOVES = 10_000
POPULATION = 600


def init():
    moves = [random.randint(0, 1) for i in range(MOVES)]
    return moves


def fitness_func(coord1, coord2, incentive_score):
    dist = Vector.dist(coord1, coord2)

    score = 1/(dist**2) + 1000*incentive_score

    return score


def make_babies(best_moves, mr):
    new_moves = []

    num_moves = len(best_moves)

    for n in range(POPULATION):
        moves = best_moves.copy()
        num_change = int(len(best_moves) * (mr/100))
        for l in range(num_change):
            r = random.randint(0, len(moves) - 1)
            moves[r] = not moves[r]

        new_moves.append(moves)
    new_moves[0] = best_moves

    for elem in new_moves:
        for n in range(MOVES - num_moves):
            elem.append(random.randint(0, 1))

    return new_moves


def next_evo(lis_cars: list, mr: int):
    best = None

    def get_best():
        nonlocal best
        idx = fitness_scores.index(max(fitness_scores))
        best = lis_cars[idx]

    # ----------- fintness scores ----------- #
    fitness_scores = []

    for elem in lis_cars:
        incent_tuple = elem.incent[0]
        car_coord = (elem.x, elem.y)
        fitness_scores.append(fitness_func(coord1=car_coord,
                                           coord2=incent_tuple,
                                           incentive_score=elem.score))

    get_best()

    all_moves = make_babies(best.moves[0:best.idx], mr)
    return all_moves


def idk():
    pass
