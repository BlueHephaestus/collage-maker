"""
The main function for this file is generate_collage(), however it also makes use of numerous other helper functions.
This will allow you to create a grid of combined images,
    where each image is of the same size as the other.
However it's worth noting that the images can be rectangles of any size (within reason), 
and the collage can be a rectangle of any size (in characters) (within reason).

I may also add more stuff to this if I wanna try it out, like:
    Step shape so that we can have overlapping images
    Different orders for how we loop through our images.

Further documentation found in each function.

-Blake Edwards / Dark Element
"""

import os,sys

import numpy as np
import cv2

def disp_img_fullscreen(img, name="test"):
    """
    Displays the given image full screen. 
    Usually used for debugging, uses opencv's display methods.
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)          
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, 1)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recursive_get_paths(directory):
    """
    Arguments:
        directory : directory to recursively traverse. Should be a string.

    Returns:
        A list of filepaths, e.g.:
            "/home/darkelement/test.txt"        
    """
    paths = []
    for (path, dirs, fnames) in os.walk(directory):
        for fname in fnames:
            paths.append(os.path.join(path, fname))
    return paths

def generate_collage(img_h, img_w, collage_h=-1, collage_w=-1, img_dir="imgs", collage_dir="collage.png"):
    """
    Arguments:
        img_h, img_w: Size to scale all individual images to in our img_dir, when making our collage.
        collage_h, collage_w: This should be given as the height and width in terms of number of IMAGES, not number of PIXELS.
            If provided, will be the resulting height and width of our collage, in terms of img_h and img_w.
            If not provided, will assume the resulting collage is to be a square,
                and will determine the collage_h and collage_w by getting the number of images in img_dir,
                via collage_h = collage_w = ceil(sqrt(img_n))
        img_dir: Directory where the images for our collage are stored. 
            We will recursively open ALL jpg and png files in this directory, so don't put this as your root directory.
            But this does mean you can have subdirectories so there.
            Note: It defaults to the imgs/ directory in the same directory as this file.
        collage_dir: Directory where to save our resulting collage.
            Defaults to collage.png in the same directory as this file.

    Returns:
        Makes an image of shape (collage_h * img_h, collage_w * img_w), and 
            loops through all images in our img_dir, 
            resizing them to shape (img_h, img_w),
            and finally inserting them into our collage image, 
            in the order left-right, top-bottom.
        Saves this image to collage_dir.
    """
    """
    First we handle our arguments.

    We handle collage_h and collage_w as documented above.
    """
    img_n = len(recursive_get_paths(img_dir))
    if collage_h < 0 or collage_w < 0:
        collage_h = collage_w = int(np.ceil(np.sqrt(img_n)))

    """
    Then we ensure our img_h and img_w are ints.
    """
    img_h, img_w = int(img_h), int(img_w)
    
    """
    Now create our image (assumes RGB)
    """
    collage = np.zeros((collage_h*img_h, collage_w*img_w, 3), dtype=np.uint8)

    """
    Now we loop through all images in our img_dir,
        resize them to our img_h, img_w,
        and insert them into our collage in the order left-right, top-bottom.
    """
    for i, img_path in enumerate(recursive_get_paths(img_dir)):
        #Print % progress
        sys.stdout.write("\r{:.2%} Complete".format(float(i)/(img_n-1)))
        sys.stdout.flush()

        #Read image
        img = cv2.imread(img_path)

        #Check to make sure we have an image, if not continue to the next path
        try:
            img.size
        except:
            continue

        #Resize
        img = cv2.resize(img, (img_h, img_w))

        #Get character row and column index via our handy equations
        row_i = i // collage_w
        col_i = i % collage_w

        #Convert them to pixel indices instead of character indices
        row_i *= img_h
        col_i *= img_w

        #If we are out of the bounds of our collage, we exit.
        if row_i >= collage.shape[0] or col_i >= collage.shape[1]:
            break

        #Insert into our collage
        collage[row_i:row_i+img_h,col_i:col_i+img_w] = img
    """
    Now that we're finished, save our collage.
    """
    cv2.imwrite(collage_dir, collage)

generate_collage(150, 150)
