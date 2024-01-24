import pygame as py
import sys
import numpy as np


class Slider:
    def __init__(self, pos: tuple, lo: int, hi: int, init_val: int, line_dims: tuple, slider_dims: tuple, slider_col: tuple, line_col: tuple, v_orientation=False):
        '''
        Creates one Slider with an attached value

        Parameters:
        - pos (tuple): Coordinates (x, y) for the slider placement
        - lo, hi, init_val (int): Specifies min, max and initial values
        - line_dims, slider_dims (tuple): (length, thickness)
        - slider_col, line_col (tuple): 3 element tuple specifying RGB values 0-255
        - v_orientation (bool): Sliders are set to vertical orientation if True

        Raises:
        - ValueError: if init_val does not fall in range of lo and hi
        '''
        if not lo <= init_val <= hi:
            raise ValueError(
                f"init_val={init_val} is outside of the bounds {lo}-{hi}")

        self.hi, self.lo, self.val = hi, lo, init_val
        self.vertical = v_orientation
        self.xpos, self.ypos = pos
        l_length, l_thick = line_dims
        s_length, s_thick = slider_dims
        slid_factor = (init_val-lo) / (hi-lo)

        if self.vertical:
            self.line = py.Surface((l_thick, l_length))
            init_x = self.xpos - s_thick/2 + l_thick/2
            init_y = self.ypos + l_length*(1-slid_factor) - s_length/2
            self.slide_rect = py.Rect(init_x, init_y, s_thick, s_length)
        else:
            self.line = py.Surface((l_length, l_thick))
            init_x = self.xpos + l_length*slid_factor - s_length/2
            init_y = self.ypos - s_thick/2 + l_thick/2
            self.slide_rect = py.Rect(init_x, init_y, s_length, s_thick)

        self.line.fill(line_col)
        self.slide_surf = py.Surface(
            (self.slide_rect.width, self.slide_rect.height))
        self.slide_surf.fill(slider_col)

    def slide(self, mouse_pos: tuple):
        '''
        Moves the slider according to the given mouse position

        Parameters:
        - mouse_pos (tuple): Coordinates (x, y) of the current mouse position

        Note:
        - I think this code is a little bit verbose but I'm keeping it like this
            for readability
        '''
        if self.vertical:
            slider_pos = mouse_pos[1] - self.slide_rect.height/2
            min_y = self.ypos - self.slide_rect.height/2
            max_y = self.ypos + self.line.get_height() - self.slide_rect.height/2

            if slider_pos < min_y:
                slider_pos = min_y
            elif slider_pos > max_y:
                slider_pos = max_y

            self.slide_rect.y = slider_pos
            self.val = self.lo + ((self.hi-self.lo) / self.line.get_height()) * \
                (max_y - slider_pos)
        else:
            slider_pos = mouse_pos[0] - self.slide_rect.width/2
            min_x = self.xpos - self.slide_rect.width/2
            max_x = self.xpos + self.line.get_width() - self.slide_rect.width/2

            if slider_pos < min_x:
                slider_pos = min_x
            elif slider_pos > max_x:
                slider_pos = max_x

            self.slide_rect.x = slider_pos
            self.val = self.lo + ((self.hi-self.lo) / self.line.get_width()) * \
                (slider_pos + self.slide_rect.width/2 - self.xpos)

    def draw(self, win):
        win.blit(self.line, (self.xpos, self.ypos))
        win.blit(self.slide_surf, (self.slide_rect.x,
                 self.slide_rect.y))


class Display:
    def __init__(self, pos: tuple, img_dims: tuple, disp_dims: tuple):
        '''
        Creates a pixel display out of pygame Rects

        Parameters:
        - pos (tuple): Coordinates (x, y) for the display placement
        - img_dims (tuple): (height, width) of the original image/array
        - disp_dims (tuple): (width, height) of the display in pygame

        Note:
        - self.pixels is a 2d python list,
            where each element is a list where the 1st element is the colour of the 
            pixel and the second is that pixel's (x,y) coordinates
        - pixel width/height is calculated with integer division, so that there is 
            some overlap between pixels, otherwise pygame freaks out and it looks
            bad (I don't atually hate how it looks)
        '''

        width, height = disp_dims
        num_rows, num_cols = img_dims
        self.pixel_width = width // num_cols
        self.pixel_height = height // num_rows

        self.pixels = [
            [None for _ in range(num_cols)]
            for _ in range(num_rows)]

        running_y = pos[1]
        for y in range(num_rows):
            running_x = pos[0]
            for x in range(num_cols):
                pix_pos = (running_x, running_y)
                self.pixels[y][x] = [(0, 0, 0), pix_pos]
                running_x += self.pixel_width
            running_y += self.pixel_height

    def update_disp(self, arr):
        '''
        Updates our pixels of the display to match the given array

        Parameters:
        - arr (2D array): (height, width), 0-255

        Note:
        - This system only supports 2d arrays, 
            ie. no RGB support 
        - Again, note the structure of self.pixels
        '''
        num_rows, num_cols = arr.shape
        for y in range(num_rows):
            for x in range(num_cols):
                col_val = arr[y, x]
                self.pixels[y][x][0] = (col_val, col_val, col_val)

    def draw(self, win):
        for row in self.pixels:
            for pixel in row:
                col, pos = pixel
                surf = py.Surface((self.pixel_width, self.pixel_height))
                surf.fill(col)
                win.blit(surf, pos)


def draw_val(win, pos, text, font, text_col,  bg_RGBA):
    '''
    Draws the value given with a background display

    Parameters:
    - win: Pygame master display
    - pos (tuple): Coordinates (x, y) for the text placement
    - text (str): Value as a string
    - font: pygame font being used
    - text_col (tuple): 3 element tuple specifying the RGB values of the text 
    - bg_RGBA (tuple): 4 element tuple specifying the RGBA values of the background
    '''

    bg_surf = py.Surface((48, 25))
    val_surf = font.render(text, True, text_col)

    bg_surf.fill(bg_RGBA[:-1])
    bg_surf.set_alpha(bg_RGBA[-1])

    bg_pos = (pos[0] - 12, pos[1] - 30)
    val_pos = (bg_pos[0]+2, bg_pos[1] + 2)
    win.blit(bg_surf, bg_pos)
    win.blit(val_surf, val_pos)


def run(model):
    '''
    UI for generaring images by playing with the latent vectors

    Parameters:
    - model (keras model): Decoder model that builds images

    Steps for "decent" results:
        Note, These steps are coming from the `latent_space_inference()` function
    - Let sliders 3 and 4 be 0
    - Move the remaining sliders between values of 0-15

    Note:
    - Sliders/nodes follow a row-wise arrangement 
        (ie. 3rd node would correspond to the slider on the first row and 3rd column)
    - Can play around with colours/fonts 
    - Changing layout is a bit more involved
    '''

    # --------------- Pygame init things --------------- #
    py.font.init()
    win = py.display.set_mode((800, 500))
    fps = 60
    clock = py.time.Clock()

    # ------------------- Appearance Variables ------------------- #
    # Background colour
    bg = (0, 0, 0)

    # Latent vector text (pop-up)
    val_font = py.font.SysFont("sfcamera", 17, bold=True)
    value_text_col = (0, 0, 0)
    value_bg_rgba = (255, 255, 255, 200)

    # Sliders (the `slider` is the block that the runs along the `line`)
    line_dims = (130, 5)    # (length, thickness)
    slider_dims = (30, 20)  # (length, thickness)
    slider_col = (255, 255, 255)
    line_col = (255, 255, 255)

    # ------------------- Init Sliders & Display ------------------- #
    inp = np.random.uniform(0, 30, (1, 10))  # Latent vectors
    sliders, val_idx = [], 0
    for y in range(2):
        for x in range(5):
            sliders.append(Slider(pos=(500 + 60*x, 100 + 200*y),
                                  lo=0, hi=30, init_val=inp[0, val_idx],
                                  line_dims=line_dims, slider_dims=slider_dims,
                                  slider_col=slider_col, line_col=line_col,
                                  v_orientation=True)
                           )
            val_idx += 1

    display = Display(pos=(0, 0),
                      img_dims=(28, 28),
                      disp_dims=(500, 500))

    # ------------------- Init variables for Loop ------------------- #
    slide_lock = False
    win.fill(bg)
    outp = model.predict(inp, verbose=0)
    display.update_disp(outp[0, :, :, 0] * 255)
    display.draw(win)
    for slider in sliders:
        slider.draw(win)
    while 1:
        clock.tick(fps)
        py.display.update()

        mouse1 = py.mouse.get_pressed()[0]
        if mouse1:
            mouse_pos = py.mouse.get_pos()
            if slide_lock:
                target_silder = sliders[target_idx]  # get slider being clicked
                changing_val = target_silder.val    # get val for slider
                target_silder.slide(mouse_pos)
                inp[0, target_idx] = changing_val  # change input
                outp = model.predict(inp, verbose=0)  # get output from decoder
                display.update_disp(outp[0, :, :, 0] * 255)

                win.fill(bg)
                display.draw(win)
                for slider in sliders:
                    slider.draw(win)

                draw_val(win=win, pos=(target_silder.slide_rect.x,
                                       target_silder.slide_rect.y),
                         text="{:05.2f}".format(changing_val), font=val_font,
                         text_col=value_text_col, bg_RGBA=value_bg_rgba)

            for idx, single_slider in enumerate(sliders):
                if single_slider.slide_rect.collidepoint(mouse_pos):
                    slide_lock = True
                    target_idx = idx
        else:
            slide_lock = False
            win.fill(bg)
            display.draw(win)
            for slider in sliders:
                slider.draw(win)

        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                sys.exit()


def main():
    # Ignore this
    class Decoder:
        def __init__(self):
            pass

        def predict(self, x, verbose):
            return np.random.uniform(0, 1, (1, 28, 28, 1))

    run(Decoder())


if __name__ == "__main__":
    main()
