import numpy as np
import json
import random

MUTATION_RATE = 50


def randomize_vars():

    w1 = np.random.rand(5, 5) - 0.5
    b1 = np.zeros((5, 1))
    w2 = np.random.rand(2, 5) - 0.5
    b2 = np.zeros((2, 1))

    dict_to_be_inserted_in_the_json_located_in_computer_files = {
        "score": 0,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2.tolist(),
    }
    json_obj = json.dumps(dict_to_be_inserted_in_the_json_located_in_computer_files,
                          indent=4)
    with open("Assets/info/params.json", mode="w") as tf:
        tf.write(json_obj)


# randomize_vars()


def init():

    with open("Assets/info/params.json", mode="r") as tf:
        json_thing = json.load(tf)

    score = json_thing["score"]
    w1 = np.array(json_thing["w1"])
    b1 = np.reshape(np.array(json_thing["b1"]), (5, 1))
    w2 = np.array(json_thing["w2"])
    b2 = np.reshape(np.array(json_thing["b2"]), (2, 1))

    return mutate(w1, b1, w2, b2, score)


def save_vars(score, w1, b1, w2, b2):

    dict_heheha = {
        "score": score,
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2.tolist(),
    }

    print("score updated to ", score)
    json_thing = json.dumps(dict_heheha, indent=4)
    with open("Assets/info/params.json", mode="w") as tf:
        tf.write(json_thing)


def Relu(Z):

    return np.maximum(Z, 0)


def soft_max(Z):

    A = np.exp(Z) / sum(np.exp(Z))
    return A


def for_prop(inps, w1, b1, w2, b2):

    z1 = w1.dot(inps) + b1
    a1 = Relu(z1)

    a2 = soft_max(w2.dot(a1) + b2)

    return a1, a2


def fitness_func():

    pass


def execute(inps: list, w1, b1, w2, b2):
    inps = np.reshape(np.array(inps), (5, 1))
    # print(inps)
    # print(inps.shape)
    a1, a2 = for_prop(inps, w1, b1, w2, b2)

    if a2[0] == a2[1]:
        return a2[0][0]
    return np.where(a2 == np.max(a2))[0]


def randnum():
    return random.randint(0, 100)


def mutate(w1, b1, w2, b2, score):
    r_num = 0.5
    for row_i in range(w1.shape[0]):
        for col_i in range(w1.shape[1]):
            if randnum() < MUTATION_RATE:
                w1[row_i][col_i] = w1[row_i][col_i] + \
                    random.uniform(-r_num, r_num)

    for row_i in range(b1.shape[0]):
        for col_i in range(b1.shape[1]):
            if randnum() < MUTATION_RATE:
                b1[row_i][col_i] = b1[row_i][col_i] + \
                    random.uniform(-r_num, r_num)

    for row_i in range(w2.shape[0]):
        for col_i in range(w2.shape[1]):
            if randnum() < MUTATION_RATE:
                w2[row_i][col_i] = w2[row_i][col_i] + \
                    random.uniform(-r_num, r_num)

    for row_i in range(b2.shape[0]):
        for col_i in range(b2.shape[1]):
            if randnum() < MUTATION_RATE:
                b2[row_i][col_i] = w1[row_i][col_i] + \
                    random.uniform(-r_num, r_num)

    return w1, b1, w2, b2, score


def parent_selec(steps, w1, b1, w2, b2):
    _, _, _, _, score = init()

    print(steps, score)
    if steps > score:
        save_vars(steps, w1, b1, w2, b2)


def evolve(steps, w1, b1, w2, b2):
    parent_selec(steps, w1, b1, w2, b2)

    pass


if __name__ == "__main__":
    pass
