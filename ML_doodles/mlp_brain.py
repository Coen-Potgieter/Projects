import numpy as np
from tkinter import *
import json


# do regulisation
# -------- done --------- #
# 1. complexity
# 2. init param tech
# 3. learning rate
# 4. activation methods (output and general)
# 5. cost function
# 6. batch size
# 7. epochs
ARCHITECTURE = None
PARAM_INIT = None
LEARNING_RATE = None
ACT_FUNC = None
OUT_ACT_FUNC = None
COST_FUNC = None
BATCH_SIZE = None
EPOCHS = None


def read_settings():

    with open("Assets/brain_data/settings.json", mode="r") as tf:
        return json.load(tf)


def save_settings(data_dict):

    json_object = json.dumps(data_dict, indent=4)
    with open("Assets/brain_data/settings.json", mode="w") as tf:
        tf.write(json_object)


def init():
    global ARCHITECTURE, PARAM_INIT, LEARNING_RATE, ACT_FUNC, OUT_ACT_FUNC,\
        COST_FUNC, BATCH_SIZE, EPOCHS

    ARCHITECTURE = arc_entr.get()
    PARAM_INIT = init_params.index(init_dm.var.get())
    LEARNING_RATE = lr_scale.get()
    ACT_FUNC = act_func.index((act_func_dm.var.get()))
    OUT_ACT_FUNC = act_func.index(out_act_dm.var.get())
    COST_FUNC = cost_func.index(cost_func_dm.var.get())
    BATCH_SIZE = int(sp_batch.get())
    EPOCHS = int(sp_epoch.get())

    settings_dict = {
        "architecture": ARCHITECTURE,
        "param_init": PARAM_INIT,
        "learning rate": LEARNING_RATE,
        "act func": ACT_FUNC,
        "out act func": OUT_ACT_FUNC,
        "cost func": COST_FUNC,
        "batch size": BATCH_SIZE,
        "epoch": EPOCHS,
    }

    save_settings(settings_dict)

    window.destroy()


bg = "#272829"
white = "#FFF6E0"


class Labels:
    def __init__(self, text):
        self.lbl = Label(text=text, font=("Arial", 14, "normal"))
        self.lbl.config(bg=bg, fg=white)

    def draw(self, r, c):
        self.lbl.grid(row=r, column=c, sticky=W)


class DropMen:
    def __init__(self, items, data):

        self.var = StringVar()
        self.var.set(items[data])

        self.widg = OptionMenu(window, self.var, *items)
        self.widg.config(bg=bg, highlightthickness=0)

    def draw(self, r, c):
        self.widg.grid(row=r, column=c, sticky=E, pady=5)


data = read_settings()

window = Tk()
window.minsize(width=50, height=200)
window.config(padx=15, pady=20, bg=bg)

arc_entr = Entry(width=10, bg=bg, fg=white, highlightthickness=0)
arc_entr.insert(END, string=data["architecture"])

lbl_init = Labels("initialisation technique: ")
lbl_hl = Labels(text="Hidden layers: ")
lbl_lr = Labels("Learning Rate: ")
lbl_af = Labels("Activation Function: ")
lbl_oa = Labels("Output Activation: ")
lbl_cf = Labels("Cost Function: ")
lbl_bs = Labels("Batch Size: ")
lbl_ech = Labels("Epochs")
lbl_title = Label(text="Neural Network Settings", font=("Arial", 22, "bold"))
lbl_title.config(highlightthickness=20,
                 highlightbackground=bg, bg=bg, fg=white)

init_params = ["(-0.5, 0.5)", "Xavier Glorot"]
init_dm = DropMen(init_params, data=data["param_init"])

act_func = ["ReLU", "Sigmoid", "SiLU", "Hyperobolic Tangent", "Softmax"]
act_func_dm = DropMen(act_func, data=data["act func"])
out_act_dm = DropMen(act_func, data=data["out act func"])

cost_func = ["Mean Squared Error", "Cross Entropy"]
cost_func_dm = DropMen(cost_func, data=data["cost func"])

lr_scale = Scale(from_=0, to=0.1, orient=HORIZONTAL,
                 resolution=0.001, bg=bg, fg=white, troughcolor=bg)
lr_scale.set(data["learning rate"])

sp_batch = Spinbox(from_=0, to=500, width=5, bg=bg,
                   fg=white, highlightthickness=0, values=data["batch size"])
sp_epoch = Spinbox(from_=0, to=50, width=5, bg=bg,
                   fg=white, highlightthickness=0, values=data["epoch"])

btn_start = Button(text="Start", command=init,
                   highlightthickness=0)
btn_start.config(bg=bg, highlightbackground=bg)

lbl_title.grid(row=0, column=0, columnspan=2, sticky=N)
lbl_hl.draw(1, 0)
lbl_init.draw(2, 0)
lbl_lr.draw(3, 0)
lbl_af.draw(4, 0)
lbl_oa.draw(5, 0)
lbl_cf.draw(6, 0)
lbl_bs.draw(7, 0)
lbl_ech.draw(8, 0)

init_dm.draw(2, 1)
act_func_dm.draw(4, 1)
out_act_dm.draw(5, 1)
cost_func_dm.draw(6, 1)


arc_entr.grid(row=1, column=1, sticky=E, pady=5)
lr_scale.grid(row=3, column=1, sticky=E, pady=5)
sp_batch.grid(row=7, column=1, sticky=E, pady=5)
sp_epoch.grid(row=8, column=1, sticky=E, pady=5)
btn_start.grid(row=9, column=0, columnspan=2)

window.mainloop()


def get_empty_arr():

    nodes = ARCHITECTURE.split(",")
    nodes = list(map(int, nodes))

    weights = []
    bias = []
    for idx, elem in enumerate(nodes):
        try:
            arr = np.zeros((nodes[idx+1], nodes[idx]))
            single_bias = np.zeros((nodes[idx+1], 1))
        except IndexError:
            pass
        else:
            weights.append(arr)
            bias.append(single_bias)
    return weights, bias


def init_params():

    def glorot_xavier(p_num, c_num):
        variance = 1/(p_num + c_num)
        return ((np.random.normal(loc=0, scale=np.sqrt(variance))))

    nodes = ARCHITECTURE.split(",")
    nodes = list(map(int, nodes))

    weights, bias = get_empty_arr()

    if PARAM_INIT == 0:
        for idx, elem in enumerate(weights):
            weights[idx] = weights[idx] + \
                (np.random.rand(weights[idx].shape[0],
                 weights[idx].shape[1]) - 0.5)
            bias[idx] = bias[idx] + np.random.rand(bias[idx].shape[0], 1) - 0.5

    elif PARAM_INIT == 1:
        for idx_real, elem in enumerate(weights):

            idx = idx_real + 1

            # if idx == len(weights) + 1:
            #     break

            plus_arr = np.copy(weights[idx_real])

            for row in range(plus_arr.shape[0]):
                for col in range(plus_arr.shape[1]):
                    plus_arr[row][col] = glorot_xavier(
                        nodes[idx_real], nodes[idx])

            weights[idx_real] = weights[idx_real] + plus_arr

    return weights, bias


def save_vars(weights, bias):
    layers = []

    for idx, elem in enumerate(weights):
        whole_neuron = []

        w = weights[idx].tolist()
        b = bias[idx].tolist()

        for row, elem in enumerate(w):

            whole_neuron.append({
                "weights": w[row],
                "bias": b[row][0]
            })

        layers.append(whole_neuron)

    json_thing = json.dumps(layers, indent=4)

    with open("Assets/brain_data/vars.json", mode="w") as tf:
        tf.write(json_thing)


def read_vars():

    weights, bias = get_empty_arr()

    with open("Assets/brain_data/vars.json", mode="r") as tf:
        data_json = json.load(tf)

    for layers_idx, layers in enumerate(data_json):
        for node_idx, node in enumerate(layers):
            bias[layers_idx][node_idx, :] = node["bias"]

            for col_idx, col in enumerate(node["weights"]):
                weights[layers_idx][node_idx, col_idx] = col

    return weights, bias


def ReLU(Z):
    return np.maximum(Z, 0)


def der_ReLU(A):
    return A > 0


def sigmoid(Z):
    return (1 / (1 + np.exp(-Z)))


def der_sigmoid(A):
    return sigmoid(A) * (1 - sigmoid(A))


def SiLU(Z):
    return Z * sigmoid(Z)


def der_SiLU(A):
    return (np.exp(A) + 1 + A * np.exp(A)) / (1 + np.exp(A))**2


def hyp_tan(Z):
    return np.tanh(Z)


def der_hyp_tan(A):
    return 1 - hyp_tan(A) * hyp_tan(A)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def der_softmax(A):
    return A


def for_prop(inpts, weights, bias):
    funcs = [ReLU, sigmoid, SiLU, hyp_tan, softmax]
    act_func = funcs[ACT_FUNC]
    out_func = funcs[OUT_ACT_FUNC]

    z = []
    a = []

    for layer_idx, layer in enumerate(weights):

        if layer_idx == 0:
            dotted = inpts
        else:
            dotted = a[layer_idx-1]

        z.append(weights[layer_idx].dot(dotted) + bias[layer_idx])

        if layer_idx == len(weights) - 1:
            a.append(out_func(z[layer_idx]))
        else:
            a.append(act_func(z[layer_idx]))

    return z, a


def one_hot(a, Y):
    # Y is a 1xm where m is example
    row = (a[-1].shape[0])
    cols = Y.size

    one_hot_Y = np.zeros((row, cols))

    for i in range(cols):
        one_hot_Y[Y[i], i] = 1

    return one_hot_Y


def back_prop(inpts, z, a, w, Y):

    funcs = [der_ReLU, der_sigmoid, der_SiLU, der_hyp_tan, der_softmax]
    der_func = funcs[ACT_FUNC]
    examples = Y.size

    Y = one_hot(a, Y)

    dw = [None for i in range(len(a))]
    dz = [None for i in range(len(a))]
    db = [None for i in range(len(a))]

    for layer_idx, layer in reversed(list(enumerate(a))):

        if layer_idx == len(a) - 1:

            dz[layer_idx] = 2 * (a[layer_idx] - Y)
            dw[layer_idx] = dz[layer_idx].dot(a[layer_idx-1].T) * (1/examples)
            db[layer_idx] = np.reshape(
                (np.sum(dz[layer_idx], axis=1) * (1/examples)).T, (a[layer_idx].shape[0], 1))
        else:
            if layer_idx == 0:
                dotted = inpts
            else:
                dotted = a[layer_idx - 1]

            dz[layer_idx] = \
                w[layer_idx + 1].T.dot(dz[layer_idx+1]) * \
                der_func(z[layer_idx])

            dw[layer_idx] = dz[layer_idx].dot(dotted.T) * (1/examples)
            db[layer_idx] = np.reshape(
                (np.sum(dz[layer_idx], axis=1) * (1/examples)).T, (a[layer_idx].shape[0], 1))

    return dw, db


def update_params(w, b, dw, db):
    for layer_idx in range(len(w)):
        w[layer_idx] = w[layer_idx] - LEARNING_RATE * dw[layer_idx]
        b[layer_idx] = b[layer_idx] - LEARNING_RATE * db[layer_idx]

    return w, b


def get_prediction(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions[0:5], Y[0:5])
    return np.sum(predictions == Y) / Y.size


def gradient_desc2(X, Y):

    w, b = init_params()

    for i in range(1000):
        z, a = for_prop(X, w, b)
        dw, db = back_prop(X, z, a, w, Y)
        w, b = update_params(w, b, dw, db)

        if i % 10 == 0:
            print(f"Iteration {i}")
            predictions = get_prediction(a[-1])
            print(get_accuracy(predictions, Y))

    save_vars(w, b)


def gradient_desc(X, Y, iterr):
    m = X.shape[1]
    starting = 0

    for i in range(iterr):
        if starting + BATCH_SIZE > m:
            starting = 0

        w, b = read_vars()

        for epch_idx in range(EPOCHS):

            batch_X = X[:, starting:starting + BATCH_SIZE]
            batch_Y = Y[starting:starting + BATCH_SIZE]
            starting += BATCH_SIZE

            z, a = for_prop(inpts=batch_X, weights=w, bias=b)
            dw, db = back_prop(inpts=batch_X, z=z, a=a, w=w, Y=batch_Y)
            w, b = update_params(w, b, dw, db)

            if epch_idx % 10:
                predictions = get_prediction(a[-1])
                print(get_accuracy(predictions, batch_Y))

        save_vars(w, b)


def predictions(X, Y):

    w, b = read_vars()
    _, a = for_prop(inpts=X, weights=w, bias=b)
    A = a[-1]
    prediction = get_prediction(A)
    acc = np.sum(prediction == Y) / Y.size
    return prediction, acc


def draw_pred(X):
    w, b = read_vars()
    _, a = for_prop(inpts=X, weights=w, bias=b)
    A = a[-1]

    nodes_in_last_layer = int(data["architecture"].split(",")[-1])
    sum_pred = sum(A)
    perc = [[i, (A[i]/sum_pred)*100] for i in range(nodes_in_last_layer)]
    return (perc)


def handler(X, Y, mode, init, iters=None):
    
    if init:
        w, b = init_params()
        save_vars(w, b)

    if mode == "learn":
        # gradient_desc2(X, Y)
        gradient_desc(X, Y, iters)

    elif mode == "test":
        return predictions(X, Y)
