import imageio
import os
import pdb


subdir = 'outfromg2/out/vola/'
filenames = os.listdir('/Users/julian/src/up/spd/' + subdir)
pdb.set_trace()
images = []
for filename in filenames:
    images.append(imageio.imread(subdir + filename))
imageio.mimsave('/Users/julian/src/up/spd/movie.gif', images)