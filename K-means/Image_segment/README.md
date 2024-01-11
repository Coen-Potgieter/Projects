<h1>Applying k-means to Image segmentation</h1>
<hr/>
<ul>
  <li>
    Using k-means to find the k most similar colours in the whole image
  </li>
  <li>
    Then mapping every pixel in the image to its respective cluster
  </li>
  <li>
    Setting k=2 for example will outline the main focal objects in the image
  </li>
  <li>
    This can also be a form of compression i imagine, setting k=16 still gives you the idea of what the picture is
  </li>
  <li>
    For now, there is only euclidian distance as the measure for similarity which is most likely not the best idea. If we have 2 pixel values of say, (255,0,0) and (0,255,0), then they are equally distance away from (0,0,255).
    I imagine this cuases some inconsistencies but will explore this at some point.
  </li>
</ul>

<h4>Future ideas</h4>
  <ul>
  <li>
    I am very interested in this idea of image segmentation
  </li>
  <li>
    I found something called a "U net" which is very good at this task, it uses convolution layers and other things
  </li>
  <li>
    Going to tackle Unets at some point aswell
  </li>
</ul>
