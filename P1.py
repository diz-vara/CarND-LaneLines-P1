
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**  

# ## Import Packages

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().magic('matplotlib inline')


# ## Read in an Image

# In[2]:

#reading in an image
# I use cv2, as I'm used to it
image = cv2.imread('test_images/solidWhiteRight.png')


#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)



# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:

import math

class Point:
    def __init__(self, x, y):
        '''Defines x and y variables'''
        self.X = x
        self.Y = y
        
class Trapezia:
    def __init__(self, top, bottom, widthtop, widthbottom) :       
        self.top = top
        self.bottom = bottom
        self.widthtop = widthtop
        self.widthbottom = widthbottom
        
class CannyParameters:
    def __init__ (self, low, high):
        self.low = low
        self.high = high
        
class HoughParameters:
    def __init__ (self, rho, theta, thr, minlen, maxgap):
        self.rho = rho
        self.theta = theta * np.pi/180
        self.thr = thr
        self.minLen = minlen
        self.maxGap = maxgap
        

class GaussParameters:
    def __init__ (self, kX=3, kY=3, sigmaX=1, sigmaY=1):
        self.kernelX = kX
        self.kernelY = kY
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        
class ThrParameters:
    def __init__ (self, kx=3, ky=3, thr=15):
        self.kernelX = kx
        self.kernelY = ky
        self.threshold = thr
        
# just shows BGR image
def showBGR(img):
    if (len(img.shape) > 2) :
        b,g,r = cv2.split(img)    # get b,g,r
        rgb_img = cv2.merge([r,g,b])
        plt.imshow(rgb_img)  
    else:
        plt.imshow(img, cmap='gray')
    
#rescale image to new width (preserving proportions)    
def rescale2width(img, newWidth):
    scale = newWidth/img.shape[1]
    out = cv2.resize(img, (0,0), fx=scale, fy=scale)
    return out,scale

# restricts value to be in the range [0 1]   
def to01(x):
    if (x < 0) :
        return 0;
    if (x > 1) :
        return 1;
    return x;    
    
def maskROI(img, roi, center):
    h = img.shape[0]
    w = img.shape[1]
    topY = to01(center.Y + roi.top) * h;
    topL = to01(center.X - roi.widthtop/2) * w;
    topR = to01(center.X + roi.widthtop/2) * w;
    bottomY = to01(center.Y + roi.bottom) * h;
    #do not move bottom
    bottomL = to01(0.5 - roi.widthbottom/2) * w;
    bottomR = to01(0.5 + roi.widthbottom/2) * w;

    vertices = np.array([[(topL, topY), (topR, topY), 
                          (bottomR, bottomY), (bottomL, bottomY)]], 
                          dtype = np.int32)
    return region_of_interest(img, vertices)
    
def grayscale(img, mode):
    """Applies the Grayscale transform
    This will return an image with only one color channel
"""
    if (mode == cv2.COLOR_BGR2GRAY or mode == cv2.COLOR_RGB2GRAY):
        return cv2.cvtColor(img, mode)
    else:
        bgr = cv2.split(img)
        if (mode <0 or mode > img.shape[2]):
            mode = 0;
        return bgr[mode]

        
    
    
def canny(img, parameters):
    """Applies the Canny transform"""
    return cv2.Canny(img, parameters.low, parameters.high)
    
    

def gaussian_blur(img, p):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (p.kernelX, p.kernelY), p.sigmaX, sigmaY = p.sigmaY)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        if (channel_count > 2):
            mask = mask  + 20
            ignore_mask_color = 0
    else:
        channel_count = 1
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    if (channel_count > 2):
        masked_image = cv2.subtract(img, mask)
    else:    
        masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, hp):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, hp.rho, hp.theta*np.pi/180, hp.thr, np.array([]), minLineLength=hp.minLen, maxLineGap=hp.maxGap)
    return lines
    
   
# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# <codecell> lines filtering

def extract_lines(img, tp, gp):
     """
     extracts white regions by subtracting 
     blurred version of the input image
     and computing canny edges over the result
     """
     median = cv2.medianBlur(img,tp.kernelX)
     diff = cv2.subtract(img, median)
     #blr = gaussian_blur(diff, gp)
     out = canny(diff,cannyParameters)
     return out
     
#extends line segment to [top bottom] range
def extend_line(line, top, bottom):
    x0 = line[0]
    x1 = line[2]
    w = x1 - x0
    h = line[3] - line[1]
    if ( h == 0): 
        h = 1e-5
       
    if w > 0:
        x0 = x1 + (bottom-line[3]) * w /h
        x1 = x1 + (top-line[3])*w / h
    return [x0, bottom, x1, top]    


def filter_lines(lines, center, roi, shape):
    cX = center.X * shape[0]
    top = (center.Y) * shape[0]
    bottom = to01(center.Y + roi.bottom) * shape[0]

    #do not move bottom
    leftB = (0.5 - roi.widthbottom/2) * shape[1]
    rightB = (0.5 + roi.widthbottom/2) * shape[1]
    leftT = (center.X - roi.widthtop) * shape[1]
    rightT = (center.Y + roi.widthtop) * shape[1]

    raw_left = []
    raw_right = []

    for _line in lines:
        line = _line[0].tolist();
        extLine = extend_line(line, top, bottom);
        if (extLine[0] > leftB and extLine[2] > leftT and extLine[0] < rightB and extLine[2] < rightT):
            if ( extLine[0] < cX) :
                raw_left.append(line);
            else:
                raw_right.append(line);                
                
    return [raw_left, raw_right]   

def line_length(line):
    a = line[0]-line[2]
    b = line[1]-line[3]
    return (np.sqrt(a*a+b*b))

def average_extended_lines(lines, top, bottom):
    meanLine = np.array(4)
    sumLen = 0
    for line in lines:
        length = line_length(line)
        extLine = extend_line(line, top, bottom)
        meanLine = meanLine + np.array(extLine) * length
        sumLen = sumLen + length
    if (sumLen > 0):
        meanLine = meanLine/sumLen
    return meanLine
    
def intersection(line1, line2):
    v21 = Point(line2[0]-line1[0], line2[1] - line1[1])    
    v1 = Point(line1[2]-line1[0], line1[3]-line1[1])
    v2 = Point(line2[2]-line2[0], line2[3]-line2[1])
    
    cross = v1.X*v2.Y - v1.Y*v2.X
    if (abs(cross) < 1e-8):
        return None
        
    t1 = (v21.X * v2.Y - v21.Y * v2.X) / cross
    result = Point(line1[0] + v1.X * t1, line1[1] + v1.Y * t1)
    return result
    
# <codecell> my function
# ## Build a Lane Finding Pipeline
# 
# 
def detectLanes(img, mode=2) :
    global drawingMode;
    global lr_lines
    scaled, scale = rescale2width(img,480);
    gray = grayscale(scaled, mode);
    gray = extract_lines(gray, thrParameters, gaussParameters);
    gray = maskROI(gray, roi, center);
    #gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    lines = hough_lines(gray, houghParameters)
    o =img.copy();

    #o = maskROI(o, roi, center);


    if (not 'drawingMode' in globals()):
        drawingMode = 3
        
    if (mode == 0):
        rawRColor = [150,0,0]
        rawLColor = [0,0,150]
    else:
        rawRColor = [0,0,150]
        rawLColor = [150,0,0]

    width = gray.shape[1]
    height = gray.shape[0]
    top = to01(center.Y + roi.top) * height
    bottom = to01(center.Y + roi.bottom) * height
    avLeft = np.array(4)
    avRight = np.array(4)

    

    if (type(lines) == np.ndarray and len(lines) > 0):
        lr_lines = filter_lines(lines, center, roi, gray.shape)
        if ( len(lr_lines[0]) > 0 ):    
            if ( drawingMode & 1 ):
                draw_lines(o, (np.array([lr_lines[0]]) / scale).astype(int),rawLColor,1);                    
            avLeft = average_extended_lines(lr_lines[0], top, bottom)
        if ( len(lr_lines[1]) > 0 ):    
            if ( drawingMode & 1 ):
                draw_lines(o, (np.array([lr_lines[1]]) / scale).astype(int),rawRColor,1);                    
            avRight = average_extended_lines(lr_lines[1], top, bottom)
        
        if (avLeft.size == 4 and avRight.size == 4):        
            if (drawingMode & 2) :
                draw_lines(o, (np.array([[avLeft, avRight]]) / scale).astype(int),[0,190,0],3);                    
    
            newCenter = intersection(avLeft, avRight)
            
            #draw and update center
            if (not newCenter is None):
                cv2.drawMarker(o, (int(newCenter.X/scale), int(newCenter.Y/scale)), [0, 190, 0],
                               cv2.MARKER_CROSS, 20, 2)
                center.X = center.X * 0.5 + newCenter.X/width  * 0.5
                center.Y = center.Y * 0.5 + newCenter.Y/height * 0.5
            #print("center = ", center.X, center.Y)        
        return o, [avLeft, avRight];
    else:
        return o, [];        

    

        
# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:

import os
imageDir="test_images/"

image_list = os.listdir(imageDir)

def imreadN(N):
    image = cv2.imread(imageDir + image_list[N])
    return image
    



# <codecell>
# Build the pipeline and run your solution on all test_images. Make copies into the test_images directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

center = Point(0.5, 0.6)
roi = Trapezia(0.02, 0.55, 0.12, 0.9)
houghParameters = HoughParameters(1,1, 10, 8, 4)
gaussParameters = GaussParameters(3,3,0.5,1)
cannyParameters = CannyParameters(90,170)
thrParameters = ThrParameters(9,2,14)
mode = 2 #red in BGR 
imageDir = 'test_images/'
outDir = 'out'
RAW = 1
AVR = 2
BOTH = 3
drawingMode = BOTH


if (not os.path.isdir(outDir) ):
    os.mkdir(outDir)


for filename in image_list:
    image = cv2.imread("test_images/" + filename)
    o, lines = detectLanes(image,mode)
    print(filename)
    plt.figure(filename)
    showBGR(o)
    outname = "out/" + filename
    cv2.imwrite(outname,o)
    #input("Press Enter to continue...")
    
    





# In[5]:



# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**



# In[3]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[6]:

def process_image(image):
    # here I call detectLanes with mode == 0 (Red in RGB)
    result, lr = detectLanes(image,0)

    return result

# Let's try the one with the solid white lane on the right first ...

# In[7]:
drawingMode = BOTH
center = Point(0.5, 0.6)
white_output = 'out/white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:
drawingMode=BOTH
center = Point(0.5, 0.6)
yellow_output = 'out/yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

center = Point(0.5, 0.6)

challenge_output = 'out/extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


