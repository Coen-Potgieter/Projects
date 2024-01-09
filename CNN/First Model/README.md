<h1>Things to Note</h1>
<hr>
<ul>
	<li>
	This CNN model is made entirely from scratch using only numpy, and was done to grasp the key concepts about how a CNN works
	</li>
	<li>
	When it comes to practical use, well, its hopelessly slow, to get any semblance of how the model works I've written `Two-Conv-Layer.py` and `One-Conv-Layer.py` that implements the model on the MNIST image recognition dataset.
	</li>
	<li>
	I get promising results when using `Tanh` for my MLP output activation along with a `MSE` as my loss function, but when trying to use `Softmax` and `Categorical cross entropy` my network produces Nans
	</li>
	<li>
	I've decided to add 2 implementations, one using 1 convolutional layer and one using 2 convolution layers
	</li>
	<li>
	I could carry on trying various np.stride_tricks things to make my model use-able but I thought I should probably learn how to use libraries and stop trying to reinvent the wheel
	</li>
	<li>
	Also i think its worth noting that this was my first time truly trying to put Docstrings to good use. This allowed me to query the types and shapes of outputs of various functions much faster than usual
	</li>
  <li>
	Lastly, files in `misc` folder are just explorations/early works and things
	</li>
</ul>



