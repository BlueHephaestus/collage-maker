# Description

As the description above states, this is a small (relatively) python script I made to easily create customisable collages of images. This means that the individual images will all be the same height and width (for now) - though they can be rectangles - and put side by side into a massive new image.

The dimensions of this massive new image can be specified in the main function call (in terms of images), and so can the dimensions of each individual image. The directory to *recursively* search for images can be specified, though it defaults to the imgs/ directory contained in this repository. The destination directory to write the image can also be specified, though it also defualts to collage.png in the repository.

Below i've included the documentation found at the beginning of the generate_collage() function (our main function), and also an example call of this function:

Documentation:
```python
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
```

Example Call:
```python
generate_collage(150, 150)
```


# Installation

1. Download this repository

2. You can run the demo by doing `python generate_collage.py`, or you can open the generate_collage.py file and read the description of the generate_collage() function for further documentation on using this program/script. The explanation in the function there should be plenty to get started on making collages using my script. Just in case it isn't or you are lazy, I have included the documentation above this section of the README, in the Description section.

# Demo

I've provided a bunch of cute pictures of pokemon in the imgs/ directory, and the generate_collage.py file is prepared to run a demo. You can see the result of the demo below, and can do it yourself by running `python generate_collage.py` in a command prompt / terminal. As a sidenote, this is also what happens when you run the example call from the description.

![Collage Demo](/collage.png)

# Final Note

You can, as always, email me any questions you have. Good luck, have fun!

-Blake Edwards / Dark Element
