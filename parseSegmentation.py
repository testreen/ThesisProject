from PIL import Image
import numpy as np


'''
Find pixels in image with specified RGB value
'''
def parseSegmentation():
    im = Image.open('./P7_HE_Default_Extended_4_2_test.png')
    im.load() # required for png.split()

    width, height = im.size

    background = Image.new("RGB", im.size, (255, 255, 255))
    background.paste(im, mask=im.split()[3]) # 3 is the alpha channel

    imarray = np.array(background)
    pixel_values = np.array(imarray).reshape((width, height, 3))
    indices = np.where(np.all(imarray == (45, 216, 39), axis=-1))
    print(list(zip(indices[0], indices[1]))[1])
    #print(np.where((imarray[:,:,0] == 29) & (imarray[:,:,2] == 45)))


if __name__ == '__main__':
    parseSegmentation()
