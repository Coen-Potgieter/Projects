from turtle import Turtle

SIZE = 0.5
NUM_SEG = 10
OFFSET = SIZE * 20


class Paddle:
    def __init__(self):
        self.ob = []

        for n in range(NUM_SEG):
            seg = Turtle(shape='square')
            seg.color('white')
            seg.pu()
            seg.shapesize(SIZE)
            seg.left(90)
            seg.forward(n * OFFSET)
            self.ob.append(seg)

    def up(self, h):
        if self.ob[len(self.ob) - 1].ycor() >= h:
            pass
        else:
            for elem in self.ob:
                elem.forward(5)

    def down(self, h):
        if self.ob[0].ycor() <= -h:
            pass
        else:
            for elem in self.ob:
                elem.backward(5)

    def start(self, w, side):
        if side == 'l':
            for elem in self.ob:
                elem.left(90)
                elem.forward(w // 2 - 50)
                elem.right(90)
        elif side == 'r':
            for elem in self.ob:
                elem.right(90)
                elem.forward(w // 2 - 50)
                elem.left(90)
