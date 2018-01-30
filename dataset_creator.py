import argparse
import glob
import skimage
from skimage import io 
from skimage import transform as tf
import os


parser = argparse.ArgumentParser("Prepares a set of images to be used for trainning/testing with cycle-GAN and similar methods, ")
parser.add_argument('-r', dest='r', type=float, default=0.8, help='rate of training data')
parser.add_argument('-A', dest='A', type=str, required=True, help='path where the images from the domain A should be found')
parser.add_argument('-B', dest='B', type=str, required=True, help='path where the images from the domain B should be found')
parser.add_argument('-wi', dest='width', type=int, default=256, help='width required for the images to be created')
parser.add_argument('-he', dest='height', type=int, default=256, help='height required for the images to be created')
parser.add_argument('-o',  dest='out',type=str, default='/tmp/', help='the output folder where to put everything')


args = parser.parse_args() 
print(args)

lA = glob.glob(args.A+'scene*.png')
lB = glob.glob(args.B+'*.jpg')

# split the images in the different datasets
lA_train = lA[0:int(args.r*len(lA))]
lA_test  = lA[int(args.r*len(lA)):]
lB_train = lB[0:int(args.r*len(lB))]
lB_test  = lB[int(args.r*len(lB)):]
ll = [lA_train, lB_train, lA_test, lB_test]
letters = ['A','B','A','B']

# create the output folders
output_folders = [args.out+'trainA', args.out+'trainB', args.out+'testA', args.out+'testB']
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# fill the images resizing them if neccessary
H = args.height
W = args.width
for l, out_folder, letter in zip(ll, output_folders, letters):
    print("filling "+out_folder)
    for i,path in enumerate(l):
        img = io.imread(path)
        Hp, Wp, _ = img.shape
        if Hp > Wp:
            img = np.swapaxes(img, 0, 1)
            Hp, Wp, _ = img.shape

        if H >= W:
            h = Hp
            w = W/H*Hp
        else:
            w = Wp
            h = H/W*Wp

        # crop
        crop_h = int((Hp-h)/2)
        crop_w = int((Wp-w)/2)
        img2 = skimage.util.crop(img,[(crop_h, crop_h),(crop_w, crop_w),(0,0)])

        # resize
        img3 = tf.resize(img2, (H,W), mode = 'reflect')

        # view
        # from skimage.viewer import ImageViewer
        # viewer  = ImageViewer(img3)
        # viewer.show()

        path = out_folder + "/" + str(i+1) +"_"+letter+'.jpg'
        io.imsave(path, (img3*255).astype('uint8'))

            


