import numpy as np


def match_y(inps):
    ball_y = inps[0]
    paddle_y = inps[1]

    diff = paddle_y - ball_y

    if diff >= 0:
        return 1
    else:
        return 0


def predictpos(inps):
    

    cpos = inps[0]
    oldpos = inps[1]
    paddle_y = inps[2]

    if oldpos[0] - cpos[0] < 0:
        return 2
    
    grad = np.arctan((cpos[1] - oldpos[1]) / (cpos[0] - oldpos[0]))

    c = cpos[1] - grad*cpos[0]

    desired_loc = grad*(-390) + c

    while not -250 < desired_loc < 255:

        if desired_loc > 255:
            x_int = (255 - c) / grad
            grad *= -1
            c = 255 - grad * (x_int)

        else:
            x_int = (-250 - c) / grad
            grad *= -1
            c = -250 - grad * (x_int)

        desired_loc = grad * (-390) + c


    if paddle_y > desired_loc:
        return 1
    else:
        return 0


def main():
    pass


if __name__ == "__main__":
    main()
