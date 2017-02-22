#**Finding Lane Lines on the Road** 

##Udacity CarND P1 Writeup by Antn Varfolomeev


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image1]: ./examples/gray.png


---

### Reflection

###1. Project pipeline

Project pipeline consists of 5 steps. 

1. Image rescaling. 
Many image processing algorithms works differently on images of different scales.
For example, you may need different Gauss kernel sizes for smoothing - and you certainly will need
separate Hough algorithm parameters.
I rescale all images to width 480 pixels (preserving poportions)

2. Color conversion.
In this project I just use red channel - the resulting image has a good color contrast for white
and yellow lines

3. Lines extraction.
I didn't use Canny edge extraction: it extracts several edges on wide lines
and tends to find edges of dark patches, cracks etc.
To extract light road markings, I use difference between original image
and it's smoothed version. Good results was achieved using median blurring.
![image1]
Simple binary threshold extracts lane marking.
I apply roi mask to the thresholded image to reduce the amount of work for Hough algorithm.

4. Hough transform
Hough transform extracts line segemnts. 

5. Line filtering
This process removes lines going in wrong directions. 


First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...