from turtle import Turtle
import random



class Ball:
    def __init__(self):
        self.ob = Turtle(shape='circle')
        self.ob.pu()
        self.ob.color('white')
        self.speed = 5

        rangle = []
        for n in range(-60, 61):
            rangle.append(n)
        for n in range(120, 241):
            rangle.append(n)

        self.ob.setheading(random.choice(rangle))

    def move(self, w, h, p1, p2, sb1, sb2):

        current_h = self.ob.heading()
        r_seth1 = random.randint(-70, 70)
        r_seth2 = random.randint(110, 250)
        for elem in p1.ob:
            if self.ob.distance(elem) < 15:
                self.ob.seth(r_seth1)
                self.speed += .5

        for elem in p2.ob:
            if self.ob.distance(elem) < 15:
                self.ob.seth(r_seth2)
                self.speed += .5

        if not -h // 2 + 20 < self.ob.ycor() < h // 2 - 10:
            self.ob.setheading(current_h - 2 * current_h)

        if -w / 2 + 10 > self.ob.xcor():
            sb1.score += 1
            self.ob.hideturtle()
            return 1

        if w / 2 - 10 < self.ob.xcor():
            sb2.score += 1
            self.ob.hideturtle()
            return 1

        self.ob.forward(self.speed)
        return 0
