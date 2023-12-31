from turtle import Turtle


class Draw:
    def __init__(self):
        self.ob = Turtle()
        self.ob.pu()
        self.ob.color('white')
        self.ob.hideturtle()
        self.ob.pensize(5)
        self.score = 0

    def draw_line(self, h):
        self.ob.setpos(0, h / 2)
        self.ob.right(90)

        while self.ob.ycor() > -h / 2:
            self.ob.pd()
            self.ob.forward(20)
            self.ob.pu()
            self.ob.forward(20)

    def sb(self, h, side):
        self.ob.clear()
        if side == 'l':
            self.ob.setpos(-120, h/2 - 100)
            self.ob.write(arg=f'{self.score}', align='center', font=('Arial', 65, 'bold'))

        elif side == 'r':
            self.ob.setpos(120, h/2 - 100)
            self.ob.write(arg=f'{self.score}', align='center', font=('Arial', 65, 'bold'))

    def add_point(self):
        self.score += 1

    def draw_end(self, loser):
        self.ob.setpos(0, 200)
        self.ob.write(f'Wow bro, @player_{loser} how tf do u lose hahaha', align='center', font=('Arial', 40, 'normal'))
        self.ob.setpos(0,100)
        self.ob.write('That\'s it bud ggs', align='center', font=('Arial', 40, 'normal'))
