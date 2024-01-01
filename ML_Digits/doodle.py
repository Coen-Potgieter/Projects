import pygame as py
import PIL.Image
import numpy as np
from pygame.locals import *
import mlp_brain
import normalize


py.font.init()

FONT1 = "rockwell"
PW_FONT = py.font.SysFont(FONT1, 20, bold=True)
PERC_FONT = py.font.SysFont("Arial", 20, bold=True)
INSTR_FONT = py.font.SysFont(FONT1, 20, bold=False)

WIDTH = 800
HEIGHT = 420

WIN = None

FPS = 60

LIGHT_BLUE = "#E1F4F3"
BLACK = "#000000"
WHITE = "#FFFFFF"
BROWN = "#706C61"
YELLOW = "#E6B31E"
LIGHTER_YELLOW = "#F8D53F"
RED = "#FF5F5F"
GREEN_PASTAL = "#D2EBCD"
LIGHT_GREY = "#8E8B82"
PURPLE = "#7F5283"
VERY_DARK_BLUE = "#333C4A"
GREY = "#333333"

# ---------------- BLUES -------------- #
LIGHTEST_BLUE = "#ECF8F9"
LIGHT_BLUE = "#CEE6F3"
BLUE = "#4682A9"

BG = GREY
LINE_COL = LIGHT_BLUE
SLIDER_COL = BLUE
TEXT_COL = LIGHTEST_BLUE
GUESS_TEXT_COL = LIGHTER_YELLOW

CANVAS_W = 420
PIXEL_WIDTH = 5

P_R = 19
ALTER = False


class Slider:
    def __init__(self):
        self.line = py.Surface((130, 2))
        self.line.fill(LINE_COL)

        init_amount_slid = (P_R + 139) // 0.3
        self.slide_rect = py.Rect(init_amount_slid, 70, 30, 20)

        self.slide_surf = py.Surface(
            (self.slide_rect.width, self.slide_rect.height))
        self.slide_surf.fill(SLIDER_COL)

    def slide(self, x):
        global P_R

        if 480 < x < 480 + self.line.get_width():
            amount_slid = x - self.slide_rect.width//2
            self.slide_rect.x = amount_slid
            P_R = int(-139 + amount_slid * 0.3)

    def draw(self):
        WIN.blit(self.line, (480, 70))
        WIN.blit(self.slide_surf, (self.slide_rect.x,
                 self.slide_rect.y - self.slide_rect.height//2))


class Pixel:
    def __init__(self, x, y):
        self.pix = py.Rect(x, y, PIXEL_WIDTH, PIXEL_WIDTH)
        self.col = 0

    def change_col(self, col):
        self.col = col

    def draw(self):
        surf = py.Surface((PIXEL_WIDTH, PIXEL_WIDTH))
        surf.fill((self.col, self.col, self.col))

        WIN.blit(surf, (self.pix.x, self.pix.y))

        pass


class Text:
    def __init__(self, text):

        self.text = text
        self.surf = PERC_FONT.render(
            f"{text[1]}% -> {text[0]}", True, GUESS_TEXT_COL)

    def draw(self):
        WIN.blit(self.surf, (self.x, self.y))

    def det_pos(self, x, y):
        self.x = x
        self.y = y


def myFunc(e):
    return e[1]


def main():

    global WIN
    WIN = py.display.set_mode((WIDTH, HEIGHT))
    stats = None
    counter = 0
    guess = False

    # -------------------- Functions ------------------------ #
    def make_guess():

        img = extract_image()
        img.save("Assets/drawn_img.png")
        show_img()
        convert_img()

    def display_perc(perc):
        nonlocal stats

        perc = [[elem[0], float('%.2f' % elem[1])] for elem in perc]

        perc.sort(reverse=True, key=myFunc)

        x = 670
        running_y = 60
        stats = [[i, Text(perc[i])] for i in range(10)]

        for elem in stats:
            elem[1].det_pos(x, running_y)
            running_y += 30

    def convert_img():
        pic = PIL.Image.open("Assets/drawn_img.png")

        inp = np.array(pic)
        inp = inp.reshape(784, 1)
        if ALTER:
            inp = normalize.add_noise(inp)

        inp = inp / 255
        perc = mlp_brain.draw_pred(inp)
        display_perc(perc)

    def extract_image():
        img = PIL.Image.new(mode="L", size=(
            num_pix, num_pix), color=BLACK)
        pixels = img.load()

        for y in range(img.size[0]):
            for x in range(img.size[1]):
                pixels[x, y] = canvas[y][x].col

        img = img.resize((28, 28))

        if ALTER:
            img = normalize.scale(img)
            img = normalize.rotate_img(img)
            img = normalize.shift(img)
        return img

    def show_img():
        nonlocal display_surf
        png_img = py.image.load("Assets/drawn_img.png")
        display_surf = py.transform.scale(png_img, (200, 200))
        pass

    def draw_canvas():

        for r_idx, row in enumerate(canvas):
            for c_idx in range(len(row)):
                canvas[r_idx][c_idx].draw()

    def change_pixel(pos, col):

        ang = 0

        iters = 40
        line_iterr = 7

        for n in range(iters):

            for i in range(line_iterr):
                new_pos = (int(pos[0] + 1/line_iterr * i * P_R * np.cos(ang)),
                           int(pos[1] + 1/line_iterr * i * P_R * np.sin(ang)))

                if new_pos[0] >= 0 and new_pos[1] >= 0:

                    canvas[new_pos[1]//PIXEL_WIDTH][new_pos[0] //
                                                    PIXEL_WIDTH].change_col(col)

            ang += 2 * (np.pi) / iters

    def display_text():
        surf = PW_FONT.render(str(P_R), True, TEXT_COL)
        WIN.blit(surf, (530, 30))

        instr_text = INSTR_FONT.render(
                    f'[C] to Clear Canvas', True, YELLOW)
        WIN.blit(instr_text, (WIDTH//2 + 50 , HEIGHT - 90))

        instr_text = INSTR_FONT.render(
                    f'[S] to Predict', True, YELLOW)
        WIN.blit(instr_text, (WIDTH//2 + 90 , HEIGHT - 50))
    # ------------------------------------------------------- #

    # -------------------- Init ------------------------ #

    num_pix = CANVAS_W // PIXEL_WIDTH

    canvas = [[[None] for i in range(num_pix)]
              for i in range(num_pix)]

    running_y = 0
    for r_idx, row in enumerate(canvas):
        running_x = 0
        for c_idx in range(len(row)):
            canvas[r_idx][c_idx] = Pixel(running_x, running_y)
            running_x += PIXEL_WIDTH
        running_y += PIXEL_WIDTH
    # ------------------------------------------------------- #

    # -------------------- Variables ------------------------ #
    clock = py.time.Clock()
    pen_slider = Slider()
    display_surf = py.Surface((1, 1))
    # ------------------------------------------------------- #


    while True:

        mouse_pos = py.mouse.get_pos()
        mouse1 = py.mouse.get_pressed()[0]
        mouse2 = py.mouse.get_pressed()[2]
        
        if mouse1:
            pos = mouse_pos
            if pos[0] > CANVAS_W and (30 < pos[1] < 80):
                pen_slider.slide(pos[0])
            else:
                try:
                    change_pixel(pos, 255)
                except IndexError:
                    pass

        elif mouse2:
            pos = mouse_pos

            if not pos[0] > CANVAS_W:
                try:
                    change_pixel(pos, 0)
                except IndexError:
                    pass
        
        WIN.fill(BG)
        WIN.blit(display_surf, (450, 100))

        

        draw_canvas()
        pen_slider.draw()
        display_text()

        if not stats is None:
            for elem in stats:
                elem[1].draw()

        counter += 1
        if counter == 1000:
            counter = 0

        if guess:
            
            if counter % 100:
                make_guess()
            
            
        
        clock.tick()
        py.display.update()

        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()

            if event.type == py.KEYDOWN:

                if event.key == py.K_s:
                    guess = not guess
                if event.key == py.K_c:
                    # clears canvas
                    for r_idx, row in enumerate(canvas):
                        for c_idx in range(len(row)):
                            canvas[r_idx][c_idx].change_col(0)


if __name__ == "__main__":
    main()
