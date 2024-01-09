import numpy as np
import os
import scratch
import sys


def init_params(architecture: tuple, mode="Uniform"):
    '''
    Initializes weights and biases to be used in the MLP with random values

    Parameters:
    - architecture (tuple): Specifies the number of nodes in each layer, including the input and the output layer (input, Layer1, Layer3, ...)
    - mode (str): Specifies the type of initialisation to be used

    Returns:
    - weights, biases (list[np.ndarray]): returns a list where each element is the layer it belongs to,
        each element represents a 2d array, (node, previous node)

    Examples:

    # Example 1: architecture=(100,50,30,20) returns:
    - weights will be a 3 element list, where each element is a 2d array of sizes
        [(50,100), (30,50), (20,30)]
    - biases will be a 3 element list, where each element is a 2d array of sizes
        [(50,1), (30,1), (20,1)]
    '''

    def uniform():
        for layer in range(1, len(architecture)):
            weights.append(np.random.uniform(low=-0.5, high=0.5,
                                             size=(architecture[layer], architecture[layer-1])))
            biases.append(np.zeros((architecture[layer], 1)))

    def xavier_glorot():
        for layer in range(1, len(architecture)):
            std = np.sqrt(1/(architecture[layer-1] + architecture[layer]))
            weights.append(np.random.randn(
                architecture[layer], architecture[layer-1]) * std)
            biases.append(np.zeros((architecture[layer], 1)))

    def normal():
        for layer in range(1, len(architecture)):

            std = 0.2
            weights.append(np.random.randn(
                architecture[layer], architecture[layer-1]) * std)
            biases.append(np.zeros((architecture[layer], 1)))

    valid_modes = {
        "Uniform": uniform,
        "X&G": xavier_glorot,
        "Normal": normal
    }

    if mode not in valid_modes.keys():
        raise ValueError(
            f"Inputed mode=`{mode}` is not valid, Valid modes are {list(valid_modes.keys())}")

    weights = []
    biases = []

    valid_modes[mode]()
    return weights, biases


def save_params(weights: list, biases: list, file_path: str):
    '''
    Saves Paramaters in a .npy file in the given file path

    Parameters:
    - weights (list[np.ndarray]): Weights used in the MLP
    - biases (list[np.ndarray]): Biases used in the MLP
    - file_path (str): path to be saved to
    '''

    def check_valid_folders():
        '''
        Checks if file path exists, if it doesnt then creates it
        '''
        if not os.path.exists(file_path):
            os.mkdir(file_path)

    def clear_folder():
        '''
        Clears folder before populating it, 
            (This was done beacuse if there were weights and biases saved before 
            that had 3 layers, then saving weights and biases with 2 layers would 
            not overwrite the 3rd layer, and when loading the weights and biases 
            later would cause errors)
        '''

        files = os.listdir(file_path)

        for file_name in files:
            os.remove(f"{file_path}/{file_name}")

    check_valid_folders()
    clear_folder()

    num_layers = len(weights)

    for layer in range(num_layers):
        np.save(f"{file_path}/weights-{layer}.npy", weights[layer])
        np.save(f"{file_path}/biases-{layer}.npy", biases[layer])


def load_params(file_path: str):
    '''
    Saves Paramaters in a .npy file in the given file path

    Parameters:
    - file_path (str): path to be saved to

    Returns:
    - w (list[2D array])
    - b (list[1D array])

    Raises:
    ValueError: If the file path cannot be found
    '''

    def get_layers():
        '''
        fetches a list of the files and counts the amount of files to get the number of layers

        Raises:
        - Value Error if there are no files in the path
        '''
        files_list = os.listdir(file_path)
        num_layers = len(files_list)//2

        if num_layers <= 0:
            raise ValueError(f"Their are no files in '{file_path}'")
        return num_layers

    if not os.path.exists(file_path):
        raise ValueError(f"File path '{file_path}' does not exist")

    num_layers = get_layers()

    w = []
    b = []
    for layer in range(num_layers):

        w.append(np.load(f"{file_path}/weights-{layer}.npy"))
        b.append(np.load(f"{file_path}/biases-{layer}.npy"))

    return w, b


def for_prop(inp: np.ndarray, w: list, b: list, act_func: str, out_func: str):
    '''
    Performs a single forward pass given an input, weights and biases

    Parameters:
    - inp (2D array): This must a column vector (nodes, examples)
    - w (list[2d array]): Weights to be used in the forward pass
    - b (list[1D array]): Biases to be used in the forward pass
    - act_func, out_func (str): See Docstring for apply_activation()

    Returns:
    - z (list[2D array]) Activations of each node before the a function was applied to it
    - a (list[2D array]) Activations of each node
    *Each element represents the layer and the arrays are column arrays - (number of nodes, examples)*
    '''

    num_layers = len(w)
    a = []
    z = []

    for layer in range(num_layers):

        # if we are on our first layer, then the inputs are to be dotted, otherwise we dot the prevoius activation layer
        if not layer:
            vctr2 = inp
        else:
            vctr2 = a[layer-1]

        z.append(w[layer].dot(vctr2) + b[layer])

        # last activation layer gets the output activation function
        if layer == num_layers - 1:
            a.append(apply_activation(z[layer], out_func))
        else:
            a.append(apply_activation(z[layer], act_func))
    return z, a


def apply_activation(inp: np.ndarray, act_func: str, derivative=False):
    '''
    Applies a speicified activation function to an array input

    Parameters:
    - inp (2D Array): Function will be applied to every element in this array
    - act_func (str): Specifies the activation function to be used
    - derivative (bool): Specifies whether to perform the derivative if the function or not

    Returns:
    - (2D Array)

    Raises:
    - ValueError: If an invalid activation function is given

    Note:
    - Valid act_func inputs are as follows:
        ["Leaky ReLU", "ReLU", "Tanh", "Sigmoid", "Softmax", "SiLU"]
    '''

    def ReLU(x):
        if derivative:
            return x > 0
        return np.maximum(x, 0)

    def hyp_tan(x):
        if derivative:
            return 1 - np.tanh(x) * np.tanh(x)
        return np.tanh(x)

    def leaky_ReLU(x, alpha=0.01):
        if derivative:
            return np.where(x >= 0, 1, alpha)
        return np.maximum(alpha * x, x)

    def sigmoid(x):
        if derivative:
            return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
        return (1 / (1 + np.exp(-x)))

    def softmax(x):
        '''
        Forward:
            Softmax(zi) = exp(zi) / sum(exp(zj))

        - This gives a sort of distribution to the last layer
        - Each node has a value between 0-1 
        - The sum of all the nodes in the layer are equal to 1

        Backward:
        
        The derivative changes when looking at different cases, i==j and i !=j
        --------------------- Chat GPT explanation i and j ------------------
        
        - i: Index representing a specific class (e.g., class 1, class 2, ..., class N).
        - j: Another index representing a different class.

        So, when you see expressions involving i and j in the context of the softmax 
        derivative, it usually means that you are considering the interactions 
        between different classes. For example:

        - softmax'(i) / z'(i): The partial derivative of the softmax output for 
        class i with respect to the pre-activated value z(i) 
        (for the same class i).

        - softmax'(i) / z'(j): The partial derivative of the softmax output for 
        class i with respect to the pre-activated value z(j) 
        (for a different class j).
        ---------------------------------------------------------------------

        - So depending on what node we are computing the calculations change.
        - This makes things complicated so the implenation below is a special case
        - This case is when we are using `CEL` as out cost function
        - This works becasue the only values that aren't 0 is the case when i==j

        To Calrify if we are using something like `MSE` then this won't work
        '''
        if derivative:
            tmp = np.exp(x)
            s = tmp / np.sum(tmp, axis=0)
            return s*(1-s)

        tmp = np.exp(x)
        return tmp / np.sum(tmp, axis=0)

    def SiLU(x):
        if derivative:
            return (np.exp(x) + 1 + x * np.exp(x)) / (1 + np.exp(x))**2
        return x * sigmoid(x)

    valid_funcs = {
        "Leaky ReLU": leaky_ReLU,
        "ReLU": ReLU,
        "Tanh": hyp_tan,
        "Sigmoid": sigmoid,
        "Softmax": softmax,
        "SiLU": SiLU
    }

    if act_func not in valid_funcs.keys():
        raise ValueError(
            f"Invalid Activation Function `{act_func}`, Valid Functions are{list(valid_funcs.keys())}")

    return valid_funcs[act_func](inp)


def one_hot(Y: np.ndarray, num_classes: int):
    '''
    One-hot encodes true labels

    Parameters:
    - Y (1D array): Each element the answer for each example
    - num_classes (int): Number of classes

    Returns:
    (2D array): (true label layer, examples)
    '''
    return np.eye(num_classes)[Y].T


def back_prop(inp: np.ndarray, z: list, a: list, w: list, Y: np.ndarray, act_func: str, out_func: str, cost_func: str):
    '''
    Performs back propagation for a unspecified batch size

    Parameters:
    - inp (2D Array): (input nodes, examples)
    - z (list[2D array]) Activations of each node before the a function was applied to it
    - a (list[2D array]) Activations of each node
    - w (list[2d array]): Weights to be used in the forward pass
    - Y (1D array): Each element the answer for each example
    - act_func, out_func (str): See Docstring for apply_activation()
    - cost_func (str): See calc_cost_grad() Docstring

    Returns:
    - dw (list[2D Array]): Gradients of the weights
    - db (list[2D Array]): Gradients of the biases

    Note:
    - For clarification of whats happening, look at ML_notes/scribbles pg7   
    - Note the appending of `a` with `inpts`
        Reason: 
            When calculating `dw` for a given layer we need to access the prevoius 
            layer's activation, but if we are on the last layer then we need to access 
            inputs so I am using the fact that list[-1] accesses the last element of the list
            Additionally, if I simply had an if statement then a network with no hidden 
            layers wouldnt work since we are going into the first if statement since its
            both the last and first layer

    - We are not computing the derivative of the output function
        (I cant seem to get it to work)
    Calculation Explanation:
    ------- General -------------
    - In back prop we are trying to find the gradient of the weights and biases
    - We do this so we know how to tweak these values in order to minimize the loss that we calculate
    - This is done by performing lots of partial derivatives
    - `dz` is the change in the change in loss due to a change in the activation of the nueron
    - similar story with `dw` and `db`
    - Exact calulations can be seen in the notes
    - But generally
        - dz[l] =   (W[l+1].T . Z[l+1]) * der_func(Z[l])
        - dw[l] =   (dz[l] . A[l]) / num_examples
        - db[l] =   sum(dz[l]) / num_examples
        Note: we are averaging the gradients over a specified amount of examples,
                this is the only reason for the `/ num_examples`
    ------- last layer ----------
    - We claculate the cost of each output node (look at calc_cost() docstring for specifics)
    - This is the gradient of our loss function with respect to the output of the layer
    - We then muliply that the derivative of our out_act function of the 
        preactivation output `z`
    - This is now our `dz` which we use for `dw` and `db`    

    ------- first layer ----------
    - The only difference here is that recall when we want to calculate `dw` we 
        want to use the activations of the prevoius layer
    - This cant be done since its the first layer so we use the activations of the inputs
        instead
    '''

    def calc_cost_grad(a, Y, cost_func):
        '''
        Calculates gradient of the loss function w.r.t preactivation of the output layer
        
        Parameters:
        - a (2D array): Last activation layer, (nodes, examples)
        - Y (2D Array): One hot encoded True labels, (nodes, examples)
        - cost_func (str): Specifies what cost function to use, valid functions are:
            ["MSE", "CEL"]

        Raises:
        - ValueError: If an invalid loss function is given


        ------------ Categorical Cross-Entropy loss -----------------
        Explanation:
        - This zeores out all the nodes where the true label is 0
        - ie. if the Y = [0,1,0,0] then the cost function, irrespective of the input
            will be [0,x,0,0]
        - Then what is x?
        - x = -ln(a[1]), becuase it is the second element where the true label = 1
        - Therefore the cost of node(i) = -Y[i] * ln(a[i])
        - So the gradient of this is -Y[i] * 1 / a[i] * a'[i]
            (The a'[i] is handled in the gradient descent step with the 
             apply_activation() function)
        
        More generally though:
        - It measures the dissimilarity between the predicted probabilities and the 
            true labels
        - Again, this means that for the nodes where the true label is 0, the 
            output cost is also zero
        - This allows us to use the implentation for out derivative softmax
        
        Example:
        - outp = [0.1, 0.2, 0.05, 0.3, 0.1, 0.15, 0.05, 0.05, 0.05, 0.05]
        - true_labels = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        returns -> [0, 1.60943791, 0, 0, 0, 0, 0, 0, 0, 0]
        -------------------------------------------------------------

        ----------------- Mean Squared Error loss -------------------
        Math:
        *Lets take a single node and single example*
        - The the MSE for this node is: (a-y)^2 
        - We want to find `dz` the gradient of the Error w.r.t the preactivated
            node value z, (the gradient is obviously just the derivative)
        - Recall z=f(a) where f() is the activation function
        - So the MSE i.t.o.z = (f(z) - y)^2
        - Now we just find the derivative = 2*(f(z)-y) * f'(z)
        - And that's what we calculate, just with all nodes in a layer with m examples
        -------------------------------------------------------------
        '''

        def mse():
            return 2*(a-Y)

        def categorical_cross_entropy():        
            return -Y * (1/a) 

        valid_modes = {
            "MSE": mse,
            "CEL": categorical_cross_entropy,
        }

        if cost_func not in valid_modes.keys():
            raise ValueError(
                f"Inputed cost_func=`{cost_func}` is not valid, Valid modes are {list(valid_modes.keys())}")

        return valid_modes[cost_func]()

    num_examples = Y.size
    num_layers = len(w)

    Y = one_hot(Y, 10)

    dz = [None for i in range(num_layers)]
    dw = [None for i in range(num_layers)]
    db = [None for i in range(num_layers)]

    # So when layer == 0 we need to access activations of prevois layer, in this case inputs
    #   so now inp gets accessed when we perform a[-1]
    appended_a = a + [inp]
    for layer in reversed(range(num_layers)):

        # last layer
        if layer == num_layers - 1:
            dz[layer] = calc_cost_grad(a[layer], Y, cost_func=cost_func) * \
                    apply_activation(inp=z[layer],
                                    act_func=out_func,
                                    derivative=True)

        # not last layer
        else:
            dz[layer] = w[layer+1].T.dot(dz[layer+1]) * \
                apply_activation(z[layer], act_func, derivative=True)

        dw[layer] = dz[layer].dot(appended_a[layer-1].T) / num_examples
        db[layer] = np.sum(dz[layer], axis=1) / num_examples

    return dw, db


def update_params(w, b, dw, db, lr):
    '''
    Updates weights and biases
    '''

    num_layers = len(w)
    for layer in range(num_layers):
        w[layer] = w[layer] - lr * dw[layer]
        b[layer] = b[layer] - lr * np.reshape(db[layer], (db[layer].size, 1))

    return w, b


def debug(**kwargs):
    '''
    - Function to help debug
    - It checks all the arrays passed for both NaN and Inf values
    '''

    for key, val in kwargs.items():
        for layer in range(len(val)):
            if np.any(np.isnan(val[layer])):
                print(f"NaN values found in {key}, layer {layer}")
                sys.exit()

        # Check for infinite values
        if np.any(np.isinf(val[layer])):
            print(f"Infinite values found in {key}, layer {layer}")
            sys.exit()


def SGD(X, Y, w, b, batch_size, steps, lr, act_func, out_func, cost_func):
    '''
    Performs Stochastic Gradient Descent

    Parameters:
    - X (2D Array): (input nodes, examples)
    - Y (1D array): Each element the answer for each example
    - w (list[2d array]): Weights to be used in the forward pass
    - b (list[1D array]): Biases to be used in the forward pass
    - batch_size (int): Number of examples to be used for each gradient descent step
    - steps (int): Number of gradient descent steps to take (iterations)
    - lr (int): Learning Rate
    - act_func, out_func (str): See apply_activation() Docstring
    - cost_func (str): See calc_cost_grad() Docstring

    Notes:
    - X must be normalized to be values between 0-1 or the gradient will explode
        (values become infinite and network will become unstable)
    - w and b cant have values that are too big, this will too casue exploding gradient
    - Can use the commented out debug() function if getting undesirable results
    - Use intellegint choices for paramters here, for example a out_func="Softmax"
        can only work with cost_func="CEL"
    '''

    tot_examples = Y.size
    batch_num = 0

    for step in range(steps):
        
        if batch_size*(batch_num+1) >= tot_examples:
            batch_num = 0
        else:
            batch_num += 1

        batch_start, batch_end = batch_size*batch_num, batch_size*(batch_num+1)
        X_batch = X[:, batch_start:batch_end]
        Y_batch = Y[batch_start:batch_end]

        z, a = for_prop(X_batch, w, b, act_func, out_func)
        dw, db = back_prop(X_batch, z, a, w, Y_batch, act_func, out_func, cost_func)

        w, b = update_params(w, b, dw, db, lr=lr)
        # debug(z=z, a=a, dw=dw, db=db, w=w, b=b)

        if step % 100 == 0:
            print(f"Iteration {step}")
            print(get_accuracy(a[-1], Y_batch))

    save_params(w, b, "Assets/MLP-Brain-Data")


def get_accuracy(a, Y):
    predictions = np.argmax(a, 0)
    print(predictions[0:5], Y[0:5])
    return np.sum(predictions == Y) / Y.size


def normalize_data(*data_sets, data_min, data_max):
    '''
    Changes every element in the arrays to 0-1 using a linear function
    
    Parameters
    - Data Sets (2D Array): (nodes, examples)
    '''

    out = []
    for idx, subset in enumerate(data_sets):
        if not ((np.all(subset >= data_min)) & (np.all(subset <= data_max))):
            raise ValueError(
                f"Set-{idx+1} has elements outside of the range {data_min}-{data_max}")
        out.append(subset / data_max)

    return out


def main():

    def reminder_of_how_dot_product_works():
        '''
        - array is 4x3
        - acts is 3x1
        - result is (4x3).dot(3x1) = (4x1)
        '''

        array = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9],
                          [10, 11, 12]])

        acts = np.array([[5], [6], [2]])
        print(array.dot(acts).T)

        print([1*5 + 2*6 + 3*2],
              [5*4 + 5*6 + 6*2],
              [7*5 + 8*6 + 9*2],
              [10*5 + 11*6 + 12*2])

    def test_init_param(arch):
        w, b = init_params(arch)
        print(len(w), len(b))
        print()
        for layer in range(len(w)):
            print(w[layer].shape)
        print()
        for layer in range(len(w)):
            print(b[layer].shape)
        print()
        print(b[-1])

    def test_save_get_params(path):
        w, b = init_params((100, 50, 30, 20))
        save_params(w, b, path)
        loaded_w, loaded_b = load_params(path)

        for layer in range(len(w)):
            print(np.allclose(w[layer], loaded_w[layer]))
            print(np.allclose(b[layer], loaded_b[layer]))

    def test_forward_prop():
        inp = np.random.rand(100, 100)
        struct = (100, 30, 20, 5)
        w, b = init_params(struct)
        z, a = for_prop(inp, w, b, "Leaky ReLU", "Softmax")
        print("z", "a\n")
        print(len(z), len(a))
        for layer in range(len(z)):
            print(z[layer].shape, a[layer].shape)

        print(z[-1][:, 0].T)
        print(a[-1][:, 0].T)

    def test_back_prop():
        inp = np.random.rand(100, 100)
        struct = (100, 10)
        w, b = init_params(struct)
        z, a = for_prop(inp, w, b, "Leaky ReLU", "Softmax")
        Y = np.random.randint(0, 10, size=100)
        dw, db = back_prop(inp, z, a, w, Y, "Leaky ReLU", "Softmax")

        print(dw[-1].shape)
        print(db[-1].shape)

    # test_init_param((100,50,30,20))
    # test_save_get_params("Assets/MLP-Brain-Data")
    # reminder_of_how_dot_product_works()
    # test_forward_prop()
    # test_back_prop()

    X_train, Y_train, X_dev, Y_dev = scratch.import_data(small=False)

    X_train, X_dev = normalize_data(X_train, X_dev, data_min=0, data_max=255)

    def learn():
        struct = (784, 10)
        w, b = init_params(struct, "X&G")
        SGD(X_train, Y_train, w, b, 100, 10_000, 0.001, "Leaky ReLU", "Softmax", "CEL")

    def test():
        w, b = load_params("Assets/MLP-Brain-Data")
        z, a = for_prop(X_dev, w, b, "Leaky ReLU", "Softmax")
        print(get_accuracy(a[-1], Y_dev))


    # learn()
    test()


if __name__ == "__main__":
    main()
