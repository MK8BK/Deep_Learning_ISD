from typing import Callable, Iterable, Optional
from PIL import Image
from matplotlib.figure import Figure
from math import ceil
import inspect

import numpy as np

def image_grid(
    images: Iterable[Image.Image], columns: int = 5, titles: Optional[Iterable] = None
) -> Figure:
    """
    Return a figure holding the images arranged in a grid

    Optionally the number of columns and/or image titles can be provided.

    Example:

         >>> image_grid(images)
         >>> image_grid(images, titles=[....])

    """
    rows = ceil(1.0 * len(images) / columns)
    fig = Figure(figsize=(10, 10.0 * rows / columns))
    if titles is None:
        titles = range(len(images))
    for k, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(rows, columns, k + 1)
        ax.imshow(img)
        ax.tick_params(axis="both", labelsize=0, length=0)
        ax.grid(b=False)
        ax.set_xlabel(title, labelpad=-4)
    return fig


def image_columns(imgs):
    nimgs = [np.array(img) for img in imgs]
    return Image.fromarray(np.vstack(nimgs))
def image_rows(imgs):
    nimgs = [np.array(img) for img in imgs]
    return Image.fromarray(np.hstack(nimgs))

def thresh_image(img, threshold=200):
    return img.convert('L').point( lambda p: 0 if p > threshold else 255 )


def get_lines(img, min_line_height=28):
    """
        Returns a list of pil images, the decomposed lines of
        the image
    """
    #margin = min_line_height//5
    nimg = np.array(img)/255.0
    line_imgs = []
    mask = np.sum(nimg, axis=1) > 0
    mask = mask.astype(int)
    keep = list(np.where(mask == 1))[0]
    if keep[-1]-keep[0]+1==len(keep):
        line_img = nimg[keep[0]:keep[-1], :] * 255
        line_img_pil = Image.fromarray(np.uint8(line_img)).convert('L')
        line_imgs.append(line_img_pil)
        return line_imgs
    #print(keep)
    break_here = []
    for i in range(len(keep)-1):
        if keep[i+1]!=keep[i]+1:
            break_here.append(i)
    prev = 0
    #print(break_here)
    nimg = nimg[np.sum(nimg, axis=1) > 0]
    for break_line in break_here:

        
        line_img = nimg[prev:break_line, :] * 255

        #else:
        #    line_img = nimg[prev-margin:break_line+margin, :] * 255

        if line_img.shape[0]>min_line_height:
            line_img_pil = Image.fromarray(np.uint8(line_img)).convert('L')
            line_imgs.append(line_img_pil)

        prev = break_line

        if break_line==break_here[-1] and nimg.shape[0]-break_line>min_line_height:
            #line_img = nimg[prev-margin:break_line+margin, :] * 255
            line_img = nimg[prev:, :] * 255
            line_img_pil = Image.fromarray(np.uint8(line_img)).convert('L')
            line_imgs.append(line_img_pil)

    return line_imgs

def get_letters(line_img, min_letter_height=28):
    line_img = line_img.rotate(90, Image.NEAREST, expand = 1)
    rot_imgs = get_lines(line_img, min_line_height=min_letter_height)
    imgs = [img.rotate(270, Image.NEAREST, expand=1) for img in rot_imgs]
    return list(reversed(imgs))


def add_margin(image=5, top=5, right=5, bottom=5, left=5):
    width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height), 0)
    result.paste(image, (left, top))
    return result


def resize_img(img, size=(28,28)):
    w, h = size[0]-12, size[1]-2
    #print(w, h)
    img = img.resize((w,h))
    img = add_margin(img, 1, 6, 1, 6)
    return img



def resize_images(imgs, size=(28,28)):
    return [resize_img(img, size=size) for img in imgs]




if __name__=="__main__":
    print(f"Empty main in : '{__file__[-12:]}'")