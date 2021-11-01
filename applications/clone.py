from matplotlib import pyplot as plt
import isce
import isceobj
import isceobj.Image.IntImage as IntImage
import isceobj.Image.SlcImage as SLC
from isceobj.Image import createImage


# load image

og_path = '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/phase_history/20200712/20200712_coherence.int'
output = og_path.replace('coherence', 'coherence_cloned')

im = createImage()
im.load(og_path + '.xml')
mm = im.memMap()

data = mm[100:200, 100:200]

im2 = im.clone()
im2.setWidth(data.shape[1])
im2.setLength(data.shape[0])
im2.setAccessMode('write')
im2.filename = output
im2.createImage()

im2.dump(output + '.xml')
data.tofile(output)
