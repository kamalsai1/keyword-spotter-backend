from __future__ import division, print_function, unicode_literals
import numpy as np
import os, sys, code, datetime
import shutil
import struct
import numpy.matlib


def htkread(Filename):

    fid=open(Filename,'rb')
    header = fid.read(12)
    try:
        (htk_size, htk_period, vec_size, htk_kind) = struct.unpack('>iihh', header) #big endean data format
        data = numpy.fromfile(fid, dtype='f')
        param = data.reshape((htk_size, int(vec_size / 4))).byteswap()
    except:
          print(Filename)
          code.interact(local=locals())

    return param


def writehtk(filename, data, fp):

    htk_size, fperiod, fdim, paramKind = np.size(data, axis=0), int(np.round(fp*1.E7)), (np.size(data, axis=1) *4), 6

    fid=open(filename,'wb')
    fid.write(struct.pack(">iihh", htk_size, fperiod, fdim, paramKind)) # ">" big endian
    np.array(data, dtype="f").byteswap().tofile(fid)
    fid.close()

