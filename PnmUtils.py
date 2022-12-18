from PIL import Image
import numpy as np

def show_pnm(file_name):
    with open(file_name, 'r') as f:
        vals = f.read().split()
    w = int(vals[1])
    h = int(vals[2])
    if vals[0] == 'P2':
        pixels = np.array(vals[4:], dtype=np.uint8).reshape((h, w))
    elif vals[0] == 'P3':
        pixels = np.array(vals[4:], dtype=np.uint8).reshape((h, w, 3))
    else:
        return None
  
    return Image.fromarray(pixels)

def show_matrix(matrix_file):
    m = np.loadtxt(matrix_file)
    m = m + m.min()
    m = m / m.max()
    m = m * 255
    m = m.astype(np.uint8)
    return Image.fromarray(m)