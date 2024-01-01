from tkinter import *
import numpy as np
import pandas as pd
import time
from PIL import Image, ImageTk, ImageChops
import random
import tkmacosx

BG = "#BBE1FA"
WHITE = "#FFFFFF"
BLACK = "#000000"


class MyButton:
    def __init__(self, text, cmd):
        self.wid = tkmacosx.Button()
        self.wid.configure(width=840//3, text=text, command=cmd)

    def draw(self, r, c):
        self.wid.grid(row=r, column=c, sticky="w")


def init():

    data = pd.read_csv("Assets/train.csv")
    data = np.array(data)
    m, n = data.shape

    # shuffle before splitting into dev and training sets

    data = data.T
    X_all = data[1:n]
    Y_all = data[0].T

    return X_all, Y_all


def fetch(num):

    data = pd.read_csv("Assets/train.csv")
    data = np.array(data)
    m, n = data.shape

    data_dev = data[0:1000].T
    X_dev = data_dev[1:n]

    data = pd.read_csv("Assets/altered_digits.csv")
    data = np.array(data)
    m, n = data.shape

    data_dev = data[0:1000].T
    X_dev_a = data_dev[1:n]
    Y_dev_a = data_dev[0].T

    return X_dev[:, num], X_dev_a[:, num], Y_dev_a[num]


def load_from_X(X):

    X_mat = X.reshape((28, 28))
    img = Image.fromarray(np.uint8(X_mat))
    return img


def convert_image(img):
    tk_img = ImageTk.PhotoImage(img)
    tk_img = tk_img._PhotoImage__photo.zoom(15, 15)

    return tk_img


def rotate_img(img):
    r_theta = random.uniform(-30, 30)
    img = img.rotate(r_theta)
    return img


def shift(img):

    rx = random.randint(-3, 3)
    ry = random.randint(-3, 3)

    return ImageChops.offset(img, rx, ry)


def add_noise(X):
    cha = 75
    amount_of_noise = random.randint(0, 100)
    for i in range(amount_of_noise):
        r_change = random.randint(-cha, cha)
        # r_change = 40

        r_idx = random.randint(0, 783)
        new_col = X[r_idx] + r_change

        if new_col > 255:
            new_col = 255
        elif new_col < 0:
            new_col = 0

        X[r_idx] = new_col

    return X


def alter_img():

    X, Y = init()

    new_dict = {
        "label": []
    }
    for n in range(784):

        new_dict[f"pixel{n}"] = []

    # print(X.shape[1])
    for ex_idx in range(X.shape[1]):

        # print(ex_idx)
        changed_X = X[:, ex_idx].copy()
        changed_X = add_noise(changed_X)

        altered_img = load_from_X(changed_X)

        altered_img = scale(altered_img)
        altered_img = rotate_img(altered_img)
        altered_img = shift(altered_img)

        new_X = convert_to_inp(altered_img)

        i = 0
        new_dict["label"].append(Y[ex_idx])
        for row in range(new_X.shape[0]):
            for col in range(new_X.shape[1]):

                new_dict[f"pixel{i}"].append(new_X[row][col])
                i += 1

    data = pd.DataFrame(new_dict)

    data.to_csv('Assets/altered_digits.csv')


def convert_to_inp(img):

    X = np.array(img)

    return X


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


def scale(img):

    r_new = random.uniform(0.8, 1.2)
    img = zoom_at(img, 16, 16, r_new)
    return img


def main():

    start = time.time()
    # alter_img()
    end = time.time()

    print(end - start)
    X, Xa, Ya = fetch(80)

    print(Ya)

    img = load_from_X(X)

    altered_img = load_from_X(Xa)

    win = Tk()

    c_all = Canvas(width=840, height=420, bg=BLACK)

    altered_img_tk = convert_image(altered_img)
    og_image_tk = convert_image(img)

    c_all.create_image(210+420, 210, image=altered_img_tk)
    c_all.create_image(210, 210, image=og_image_tk)

    c_all.grid(row=0, column=0, columnspan=3)

    win.mainloop()


if __name__ == "__main__":
    main()
