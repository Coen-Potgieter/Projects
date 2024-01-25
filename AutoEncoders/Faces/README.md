<h1>Face Generation</h1>

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
		<h5>`generate_faces()`</h5> This lets the user generate faces by playing around with the latent space using sliders
		</li>
		<li>
		<h5>`draw_faces()`</h5> Again generated faces, but this time from the users sketchings
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
		The data used for this project is a collection of high school graduate pictures from the 1930-2013
	</li>
	<li>
		There are both Male and Female pictures, and are all grayscale
	</li>
	<li>
		Original set can be found here: https://github.com/p-lambda/gradual_domain_adaptation
	</li>
	<li>
		Numpy files only contain a small portion of the dataset as to keep everything lightweight 
	</li>
</ul>

<br/>
<h3>Note</h3>
<ul>
	<li>
		`data_prep_understand.py` is a script that helped me prepare the data and convert it to be usable for training, it has little functionality now
	</li>
</ul>