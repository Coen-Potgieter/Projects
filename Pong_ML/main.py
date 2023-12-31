from turtle import Turtle, Screen
from paddles import Paddle
from Ballas import Ball
from drawer import Draw
import time
import brain

WIDTH = 900
HEIGHT = 525


def main():
    ms = Screen()
    ms.bgcolor('black')
    ms.tracer(0)
    ms.setup(width=WIDTH, height=HEIGHT)

    line = Draw()
    line.draw_line(HEIGHT)

    sb1 = Draw()
    sb2 = Draw()

    p1 = Paddle()
    p1.start(WIDTH, 'l')
    p2 = Paddle()
    p2.start(WIDTH, 'r')
    dict_move = {
        'key2up': 0,
        'key2down': 0,
    }

    ball = Ball()

    def keydown2():
        dict_move['key2up'] = 1

    def keyup2():
        dict_move['key2up'] = 0

    def keydown2d():
        dict_move['key2down'] = 1

    def keyup2d():
        dict_move['key2down'] = 0

    prev_ball_pos = ball.ob.pos()
    run = 1

    while run:

        sb1.sb(HEIGHT, 'r')
        sb2.sb(HEIGHT, 'l')

        if ball.move(w=WIDTH, h=HEIGHT, p1=p1, p2=p2, sb1=sb1, sb2=sb2):
            ball = Ball()

        # -------------------------- implementation 1 -------------------------- #
        """ Match Y """
        # inputs = [ball.ob.pos()[1],
        #           p1.ob[4].pos()[1]]

        # action = brain.match_y(inputs)
        # if action:
        #     p1.down(HEIGHT//2)
        # else:
        #     p1.up(HEIGHT//2)
        # ----------------------------------------------------------------------- #

        # -------------------------- implementation 2 -------------------------- #
        choice = brain.predictpos(
            [ball.ob.pos(), prev_ball_pos, p1.ob[4].pos()[1]])

        if choice == 2:
            pass
        elif choice:
            p1.down(HEIGHT//2)
        else:
            p1.up(HEIGHT//2)

        prev_ball_pos = ball.ob.pos()

        # ----------------------------------------------------------------------- #

        ms.listen()

        ms.onkeypress(key='Up', fun=keydown2)
        ms.onkeyrelease(key='Up', fun=keyup2)
        ms.onkeypress(key='Down', fun=keydown2d)
        ms.onkeyrelease(key='Down', fun=keyup2d)

        if dict_move['key2up']:
            p2.up(HEIGHT // 2)

        if dict_move['key2down']:
            p2.down(HEIGHT // 2)

        ms.update()

    if sb1.score > sb2.score:
        lose = 1
    elif sb1.score < sb2.score:
        lose = 2
    else:
        lose = None

    time.sleep(1)
    ms.clearscreen()
    ms.tracer(0)
    ms.bgcolor('black')
    end = Draw()

    end.draw_end(lose)

    ms.exitonclick()


if __name__ == '__main__':
    main()
