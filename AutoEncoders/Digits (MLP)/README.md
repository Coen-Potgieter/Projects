<h1>Digit Generation</h1>

<hr/>
<h3>How to Use</h3>
<ul>
	<li>
		Execute `main.py`
	</li>
	<li>
		Can comment in/out the functions that you want run
	</li>
	<li>
		The following functions are as follows:
	</li>
		<ol>
		<li>
		<h5>`generate_digits()`</h5> This lets the user generate digits by playing around with the latent space using sliders
		</li>
		<li>
		<h5>`test_auto()`</h5> Showcases the inputs vs the Reconstructions made by the auto-encoder
		</li>
		<li>
		<h5>`latent_space_inference()`</h5> Gives an intuition for how the latent vectors are distributed by providing plots and statistics about the data
		</li>
		</ol>
</ul>
<br/>
<h3>Data</h3>
<ul>
	<li>
		The data used for this project is the classic MNIST handwritten digits dataset
	</li>
	<li>
		Dataset is pulled from the `keras` library and loaded onto system when program is run 
	</li>
</ul>

<br/>
<h3>Note</h3>
<ul>
	<li>
		This auto-encoder is fully connected, ie. There are no convolutional layers
	</li>
	<li>
		When the program is first run make sure that in the function `load_data()` the parameter `import_data` is set to True. This will download the dataset and save to the system. Once the dataset is downloaded you can set this parameter to False again
	</li>
</ul>
