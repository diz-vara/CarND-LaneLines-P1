#**Finding Lane Lines on the Road** 

##Udacity CarND P1 Writeup by Anton Varfolomeev


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---
**Results (images and videos with lane marking) are in the ./out directory**
---


[//]: # (Image References)
[image1]: ./examples/gray.png
[image2]: ./examples/diff.png
[image3]: ./examples/result.png


---

### Reflection

###1. Project pipeline

Project pipeline consists of 5 steps. 

####1. Image rescaling. 
Many image processing algorithms works differently on images of different scales.
For example, you may need different Gauss kernel sizes for smoothing - and you certainly will need
separate Hough algorithm parameters.
I rescale all images to width 480 pixels (preserving proportions)

####2. Color conversion.
In this project I just use red channel - the resulting image has a good color contrast for white
and yellow lines
![image1]

####3. Lines extraction.
I didn't use Canny edge extraction: it extracts several edges on wide lines
and tends to find edges of dark patches, cracks etc.
To extract light road markings, I use difference between original image
and it's smoothed version. Good results was achieved using median blurring.

![image2]

Simple binary threshold extracts lane marking.
I apply ROI mask to the thresholded image to reduce the amount of work for Hough algorithm.

####4. Hough transform
Hough transform extracts line segments. 

####5. Line filtering
This process removes lines going in wrong directions. 
To do it, I extend each line segment to the top and bottom ROI boundaries, 
and preserve only segments which cross top and bottom sides of the ROI's 
trapezium.

####6. Line averaging
Remaining lines are separated to two groups (left and right) and averaged.
The next image shows the result with raw line segments in red and 
averaged lines shown in green.
![image3]

###2. Identify potential shortcomings
This pipeline has a number of shortcomings:
- it is extremely sensitive to the change of camera position/orientation
- it doesn't work in 'night mode'
- it doesn't work with extremely wide and short lane marking
- it will have problem with short and sparse lane marking
- it will give wrong results in case of complex marking (several lines close to each other)

###3. Possible improvements

A possible improvement would be to:
- dynamically update ROI position by vanishing point recalculation
- try another 'line extracting' algorithms (use first quartile instead of median, 
use gray erosion etc).
- use morphological analysis to remove small particles and reduce the width of the
wide ones
- use weighted average of consecutive frames to get a stable result on sparse marking
- cluster 'extended' lines into several groups for several (more than 2) lines
