import time
import numpy as np
import PIL.Image
import PIL.ImageTk
import mlp_brain
import matplotlib.pyplot as plt
from tkinter import *
import io
import tkmacosx
import doodle

BG = "#272829"
BLACK = "#000000"
WHITE = "#FFF6E0"
GREY = "#D8D9DA"

WRONG_IDX = -1

ITEMS = ["bicycle",
         "bread",
         "pizza",
         "windmill",
         "tree",
         "cloud",
         "fork",
         "sun",
         "line",
         "leaf",
         "ocean",
         "pencil",
         "star",
         "stethoscope"]


def get_data(mode):

    if mode == "dev":
        lis_of_npArrays = [
            np.load(f"Assets/data/{elem}.npy").T[:, 50_000:55_000] / 255 for elem in ITEMS]

        Y = [np.zeros((1, elem.shape[1])) + idx for idx,
             elem in enumerate(lis_of_npArrays)]

        for n in range(len(lis_of_npArrays)):
            if n == 0:
                combined_X = lis_of_npArrays[0]
                combined_Y = Y[0]
            else:
                combined_X = np.concatenate(
                    (combined_X, lis_of_npArrays[n]), axis=1)
                combined_Y = np.concatenate(
                    (combined_Y, Y[n]), axis=1)

        combined_all = np.concatenate((combined_Y, combined_X), axis=0)

        np.random.shuffle(combined_all.T)
        X_dev = combined_all[1:785, :]
        Y_dev = combined_all[0, :]

        return X_dev, Y_dev

    elif mode == "train":
        lis_of_npArrays = [
            np.load(f"Assets/data/{elem}.npy").T[:, 0:50_000] / 255 for elem in ITEMS]

        Y = [np.zeros((1, elem.shape[1])) + idx for idx,
             elem in enumerate(lis_of_npArrays)]

        for n in range(len(lis_of_npArrays)):
            if n == 0:
                combined_X = lis_of_npArrays[0]
                combined_Y = Y[0]
            else:
                combined_X = np.concatenate(
                    (combined_X, lis_of_npArrays[n]), axis=1)
                combined_Y = np.concatenate(
                    (combined_Y, Y[n]), axis=1)

        combined_all = np.concatenate((combined_Y, combined_X), axis=0)

        np.random.shuffle(combined_all.T)

        X_train = combined_all[1:785, :]
        Y_train = combined_all[0, :]

        return X_train, Y_train.astype(int)


class Buttons:
    def __init__(self, text, cmd):
        self.wid = tkmacosx.Button(
            text=text, width=250, height=60, bg=BLACK, command=cmd)
        self.wid.config(activebackground=(GREY, GREY),
                        focuscolor=GREY, bd=0,
                        borderless=True, highlightthickness=0, fg=WHITE)

    def draw(self, r, c):
        self.wid.grid(row=r, column=c)


def test(index, X=None, Y=None):

    if X is None or Y is None:
        X, Y = get_data("dev")

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
        f"Total accuarcy: {np.round(accuracy, decimals=2)}\n"
        f"prediction: {ITEMS[predict[index].astype(int)]}\n"
        f"label: {ITEMS[Y[index].astype(int)]}")

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


def train(iterations, init):
    print("start")
    start = time.time()
    X, Y = get_data("train")
    print(f"{time.time() - start}s")

    mlp_brain.handler(X, Y, mode="learn", init=init, iters=iterations)


def draw():
    doodle.main()


def main():
    # [0] -> bicycle
    # [1] -> bread
    # [2] -> pizza
    # [3] -> windmill
    # [4] -> tree
    # [5] -> cloud
    # [6] -> fork
    # [7] -> sun
    # [8] -> line
    # [9] -> leaf
    # [10] -> ocean
    # [11] -> pencil
    # [12] -> star
    # [13] -> stethoscope

    

    # test(index=0)

    draw()


if __name__ == "__main__":
    main()
