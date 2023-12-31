import pygame
import Vector as vct
import math
import Genetic_algo
import reinforcement_learn

pygame.font.init()

PI = math.pi

FONT1 = pygame.font.SysFont("Arial", 20, bold=True)


SHOW_MASK = False

WIDTH = 1000
HEIGHT = 700

CAR_W = 20
CAR_H = CAR_W / (607/301)

TRACK_W = 1000
TRACK_H = 700

LINE_W = 1000
LINE_H = 3

INCENT_W = 30
INCENT_H = INCENT_W / (938/853)
# -------------- TRACKS -------------- #
TRACK_NUM = 1
INCENTIVES_1 = [(140, 100), (340, 250),
                (160, 500), (320, 420),
                (800, 550), (850, 170),
                (WIDTH/2 - 100, 60)]

INCENTIVES_2 = [(140, 100), (340, 250),
                (160, 500), (320, 420),
                (800, 550), (850, 170),
                (WIDTH/2 - 100, 60)]


WIN = pygame.display.set_mode((WIDTH, HEIGHT))

PNG_CAR = pygame.image.load("Assets/car.png")
PNG_TRACK = pygame.image.load(f"Assets/Tracks/track{TRACK_NUM}.png")
PNG_TRACK_OUTLINE = pygame.image.load(
    f"Assets/Tracks/track{TRACK_NUM}_outline.png")
PNG_LINE = pygame.image.load("Assets/line.png")
PNG_MARKER = pygame.image.load("Assets/marker.png")
PNG_INCENT = pygame.image.load("Assets/incentive.png")
PNG_BETTER_CAR = pygame.image.load("Assets/car_best.png")

CAR = pygame.transform.scale(PNG_CAR, (CAR_W, CAR_H))
TRACK = pygame.transform.scale(PNG_TRACK, (TRACK_W, TRACK_H))
TRACK_OL = pygame.transform.scale(PNG_TRACK_OUTLINE, (TRACK_W, TRACK_H))
LINE = pygame.transform.scale(PNG_LINE, (LINE_W, LINE_H))
MARKER = pygame.transform.scale(PNG_MARKER, (10, 10))
INCENTIVE_IMG = pygame.transform.scale(PNG_INCENT, (INCENT_W, INCENT_H))
CAR_BEST = pygame.transform.scale(PNG_BETTER_CAR, (CAR_W, CAR_H))

TRACK_MASK = pygame.mask.from_surface(TRACK_OL)
CAR_MASK = pygame.mask.from_surface(CAR)

FPS = 60

GREY = "#393E46"
LIGHT_GREY = "#EAEAEA"
WHITE = "#FFFFFF"

POPULATION = Genetic_algo.POPULATION
MUTATION_RATE = 10


class Car:
    def __init__(self, moves=None, incent=None):

        self.x = WIDTH/2
        self.y = 100

        self.direc = PI
        self.acc = vct.Vector(0, self.direc)
        self.vel = vct.Vector(0, self.direc)

        if not incent is None:
            self.score = 0
            self.incent = incent

        if not moves is None:
            self.moves = moves
            self.idx = 0

        self.done = False
        self.surf = CAR

        self.best = False

    def draw(self):

        WIN.blit(self.surf, self.rect)

    def see_hitbox(self):
        surf = pygame.Surface((self.rect.width, self.rect.height))
        surf.fill(LIGHT_GREY)
        WIN.blit(surf, self.rect)

    def _rotate(self):
        if not self.best:
            surf = CAR
        else:
            surf = CAR_BEST

        self.surf = pygame.transform.rotate(
            surf, math.degrees(self.direc))

        self.rect = self.surf.get_rect(center=(self.x, self.y))

    def move_aut2(self, l_r):
        if l_r == 0:
            self.left()
        elif l_r == 1:
            self.right()

        self.forward()

    def move_aut(self):

        if self.moves[self.idx]:
            self.left()
        else:
            self.right()

        self.idx += 1
        self.forward()

    def move_man(self, pressed):
        if pressed[pygame.K_LEFT]:
            self.left()

        if pressed[pygame.K_RIGHT]:
            self.right()

        if pressed[pygame.K_UP]:
            self.forward()

    def _move(self):

        self.acc = vct.Vector(self.acc.mag, self.direc)

        if math.degrees(self.direc) > 360:
            self.direc -= 2*PI
        elif math.degrees(self.direc) < 0:
            self.direc += 2*PI

        self.vel.direction = self.direc

        # ------------- friction ------------- #

        # if self.vel.mag > 0.05:
        #     self.acc.mag -= 0.1
        # else:
        #     self.vel.mag = 0

        # ------------- updating velocity ------------- #

        self.vel += self.acc

        if self.vel.mag > 3:
            self.vel.mag = 3

        self.x += self.vel._xcomp()
        self.y -= self.vel._ycomp()

    def collide(self, mask):
        car_mask = pygame.mask.from_surface(self.surf)
        nicex = int(self.x - CAR_W/2)
        nicey = int(self.y - CAR_H/2)
        offset = (nicex, nicey)
        if mask.overlap(car_mask, offset) != None:
            self.done = True

        # WIN.blit(car_mask.to_surface(), (nicex, nicey))
        return self.done

    def left(self):
        self.direc += PI/50

    def right(self):
        self.direc -= PI/50

    def forward(self):
        self.acc += vct.Vector(magnitude=0.2, direction=self.direc)

    def score_update(self):
        if self.rect.collidepoint(self.incent[0]):
            self.incent.pop(0)
            self.score += 1

    def add_dist(self, dist):
        self.inps = dist

    def add_params(self, params):

        self.w1 = params[0]
        self.b1 = params[1]
        self.w2 = params[2]
        self.b2 = params[3]


class Eyes:
    def __init__(self):
        self.surf = LINE

    def orientation(self):

        if math.degrees(self.direc) > 360:
            self.direc -= 2*PI
        elif math.degrees(self.direc) < 0:
            self.direc += 2*PI

        self.surf = pygame.transform.rotate(
            LINE, math.degrees(self.direc))

        if 0 < self.direc <= PI/2:
            local_angle = self.direc

            self.y = self.car_y - (LINE_W * math.sin(local_angle))
            self.x = self.car_x

        if PI/2 < self.direc <= PI:
            local_angle = PI - (self.direc)

            self.y = self.car_y - (LINE_W * math.sin(local_angle))
            self.x = self.car_x - (LINE_W * math.cos(local_angle))

        if PI < self.direc <= PI * 3/2:
            local_angle = self.direc - PI

            self.y = self.car_y
            self.x = self.car_x - (LINE_W * math.cos(local_angle))

        if PI * 3/2 < self.direc <= 2*PI:
            self.x = self.car_x

            self.y = self.car_y

    def collide(self, x=0, y=0):
        line_mask = pygame.mask.from_surface(self.surf)
        offset = (int(self.x - x), int(self.y - y))
        chosen = None
        new_mask = TRACK_MASK.overlap_mask(line_mask, offset)
        mask_img = new_mask.to_surface()
        if SHOW_MASK:
            WIN.blit(mask_img, (0, 0))

        poi_lis = new_mask.outline()

        min_d = 10_000
        for elem in poi_lis:
            dist = math.sqrt((elem[0] - x)**2 + (elem[1] - y)**2)
            if dist < min_d:
                chosen = elem
                min_d = dist

        return chosen, min_d

    def update(self, x, y, direc, offset):
        self.direc = direc + offset
        self.car_x = x
        self.car_y = y

        self.orientation()

    def draw(self):

        WIN.blit(self.surf, (self.x, self.y))


class Speed:
    def __init__(self):
        self.val = 1

    def double(self):
        self.val *= 2

    def half(self):
        self.val /= 2

    def draw(self):
        if self.val >= 1:
            self.val = int(self.val)

        surf = FONT1.render(f"Speed: {self.val}x", True, WHITE)
        surf.set_alpha(100)
        WIN.blit(surf, (WIDTH - 120, 0))


def genetic_ev():

    global SHOW_MASK
    cars = None
    fps = 60
    speed = Speed()

    def next_evo():
        nonlocal cars
        am = Genetic_algo.next_evo(cars, MUTATION_RATE)
        cars = [Car(elem, incent=INCENTIVES_1.copy()) for elem in am]
        cars[0].best = True

    cars = [Car(Genetic_algo.init(), incent=INCENTIVES_1.copy())
            for i in range(POPULATION)]

    clock = pygame.time.Clock()

    show_best = False
    while True:

        num_done = 0

        WIN.fill(GREY)
        WIN.blit(TRACK, (0, 0))
        speed.draw()

        for elem in cars[0].incent:
            WIN.blit(INCENTIVE_IMG, elem)

        for elem in cars:

            if elem.collide(TRACK_MASK):
                num_done += 1

            else:
                elem.move_aut()
                elem._rotate()
                elem._move()
                if not show_best:
                    elem.draw()
                else:
                    if elem.best:
                        elem.draw()

            elem.score_update()

        if num_done == POPULATION:
            next_evo()

        clock.tick(fps)
        pygame.display.update()

        # print(cars[0].incent)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    show_best = not show_best

                if event.key == pygame.K_DOWN:
                    fps /= 2
                    speed.half()

                if event.key == pygame.K_UP:
                    fps *= 2
                    speed.double()


def manual():
    global SHOW_MASK

    car = Car()

    lines = [Eyes() for i in range(5)]
    angles = [-60, -35, 0, 30, 60]

    clock = pygame.time.Clock()

    while True:

        if not SHOW_MASK:
            WIN.fill(GREY)
            WIN.blit(TRACK, (0, 0))

        markers = []
        dist = []
        for idx, elem in enumerate(lines):
            elem.update(car.x, car.y, car.direc, math.radians(angles[idx]))
            marker_poi, marker_dist = elem.collide()
            markers.append(marker_poi)
            dist.append(marker_dist)
            elem.draw()

        for idx, elem in enumerate(markers):
            if elem != None:
                WIN.blit(MARKER, elem)
            print(f"for line {idx}, dist = {dist[idx]}")

        keys_pressed = pygame.key.get_pressed()

        car.move_man(keys_pressed)

        car._rotate()
        car._move()
        car.draw()

        if car.collide(TRACK_MASK):
            main()

        clock.tick(FPS)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:

                    SHOW_MASK = not SHOW_MASK
                    print(SHOW_MASK)


def rein_learn():
    global SHOW_MASK

    def update_params():
        car = cars[0]
        reinforcement_learn.evolve(steps, car.w1, car.b1, car.w2, car.b2)
        main()
        pass

    cars = [Car() for i in range(2)]

    lines = [Eyes() for i in range(5)]
    angles = [-60, -35, 0, 30, 60]

    clock = pygame.time.Clock()
    for elem in cars:
        w1, b1, w2, b2, _ = reinforcement_learn.init()
        params = [w1, b1, w2, b2]
        elem.add_params(params=params)

    steps = 0
    num_done = 0
    while True:
        steps += 1
        if num_done == 1:
            update_params()
        if not SHOW_MASK:
            WIN.fill(GREY)
            WIN.blit(TRACK, (0, 0))

        for elem_c in cars:
            markers = []
            dist = []
            for idx, elem in enumerate(lines):
                elem.update(elem_c.x, elem_c.y, elem_c.direc,
                            math.radians(angles[idx]))
                marker_poi, marker_dist = elem.collide()
                markers.append(marker_poi)
                dist.append(marker_dist)
                elem.draw()

            elem_c.add_dist(dist)

            for idx, elem in enumerate(markers):
                if elem != None:
                    WIN.blit(MARKER, elem)
                # val = inverse_normalization(dist[idx], 0, 0.01)
                # print(f"for line {idx}, dist = {val}")

        for elem in cars:

            right_left = reinforcement_learn.execute(elem.inps,
                                                     elem.w1,
                                                     elem.b1,
                                                     elem.w2,
                                                     elem.b2)
            elem.move_aut2(right_left)

            if not elem.done:
                elem._rotate()
                elem._move()
                elem.draw()

            if elem.collide(TRACK_MASK):
                elem.done = True
                num_done += 1

        clock.tick(FPS)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:

                    SHOW_MASK = not SHOW_MASK
                    print(SHOW_MASK)


def main():
    # genetic_ev()
    manual()
    # rein_learn()


if __name__ == "__main__":
    main()
