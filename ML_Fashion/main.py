import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageTk
import mlp_brain
import matplotlib.pyplot as plt
from tkinter import *
import io
import tkmacosx

BG = "#272829"
BLACK = "#000000"
WHITE = "#FFF6E0"
GREY = "#D8D9DA"

WRONG_IDX = -1

ITEMS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def get_data():

    data_train = pd.read_csv("Assets/fashion_train.csv")
    data_test = pd.read_csv("Assets/fashion_test.csv")

    data_train = np.array(data_train)
    data_test = np.array(data_test)

    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    data_test = data_test.T
    data_train = data_train.T

    Y_dev = data_test[0].T

    X_dev = data_test[1:785]
    X_dev = X_dev / 255.

    Y_train = data_train[0].T
    X_train = data_train[1:785]
    X_train = X_train / 255.

    return X_train, Y_train, X_dev, Y_dev


class Buttons:
    def __init__(self, text, cmd):
        self.wid = tkmacosx.Button(
            text=text, width=250, height=60, bg=BLACK, command=cmd)
        self.wid.config(activebackground=(GREY, GREY),
                        focuscolor=GREY, bd=0,
                        borderless=True, highlightthickness=0, fg=WHITE)

    def draw(self, r, c):
        self.wid.grid(row=r, column=c)


def test(index, X, Y):
    def close():
        win.destroy()

    def next_ex():
        win.destroy()
        test(index + 1, X, Y)
        # show_img(index+1, X)
        pass

    def wrong():
        global WRONG_IDX
        wrong_idx = np.where(predict != Y)
        # print(t(wrong_idx[0][0]))
        win.destroy()
        WRONG_IDX += 1
        test(wrong_idx[0][WRONG_IDX], X, Y)
        pass

    predict, accuracy = mlp_brain.handler(
        X, Y, mode="test", init=False)
    print(
        f"Total accuarcy: {accuracy}\n"
        f"prediction: {ITEMS[predict[index]]}\n"
        f"label: {ITEMS[Y[index]]}")

    current_image = X[:, index, None]
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')

    fig = plt.gcf()
    fig.set_facecolor(BLACK)

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)

    img_w = img.width
    img_h = img.height

    win = Tk()
    win.config(bg=BLACK)
    win.minsize(width=img_w, height=img_h + 50)
    w = img_w+100  # width for the Tk root
    h = img_h+70  # height for the Tk root

    # get screen width and height
    ws = win.winfo_screenwidth()  # width of the screen
    hs = win.winfo_screenheight()  # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws/2) - 700
    y = (hs/2) - 500

    # set the dimensions of the screen
    # and where it is placed
    win.geometry('%dx%d+%d+%d' % (w, h, x, y))

    tk_img = PIL.ImageTk.PhotoImage(img)

    canvas = Canvas()
    canvas.config(width=img_w, height=img_h, highlightthickness=0)
    canvas.create_image(int(img_w/2), int(img_h/2), image=tk_img)

    btn_close = Buttons("Close", close)
    btn_next = Buttons("Next", next_ex)
    btn_wrong = Buttons("Wrong", wrong)

    canvas.grid(row=0, column=0, columnspan=3, sticky=N)
    btn_close.draw(1, 0)
    btn_next.draw(1, 1)
    btn_wrong.draw(1, 2)
    win.mainloop()


def make_pic(X):
    idx = 0

    pixel_num = X[:, 0] * 255

    img = PIL.Image.new('L', (28, 28), color=0)
    i = 0
    for row in range(img.size[0]):
        for col in range(img.size[1]):

            img.putpixel((row, col), value=int(pixel_num[i]))
            i += 1

    scaled_img = img.resize((300, 300))
    scaled_img.show()
    pass


def learn(X, Y, rand_var, iters=None):

    if iters is None:
        iters = 20
    mlp_brain.handler(X, Y, mode="learn", init=rand_var, iters=iters)


def main():
    #   [0] T-shirt/top
    #   [1] Trouser
    #   [2] Pullover
    #   [3] Dress
    #   [4] Coat
    #   [5] Sandal
    #   [6] Shirt
    #   [7] Sneaker
    #   [8] Bag
    #   [9] Ankle boot

    X_train, Y_train, X_dev, Y_dev = get_data()

    # learn(X_train, Y_train, True, iters=20)

    test(0, X_dev, Y_dev)


if __name__ == "__main__":
    main()
