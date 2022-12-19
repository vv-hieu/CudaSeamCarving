from PIL import Image
import numpy as np

SEAM_COLOR = (255, 0, 255)

def show_pnm(file_name, seam = None):
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

    if (seam is not None):
        for seamPoint in seam:
            pixels[seamPoint[1]][seamPoint[0]] = SEAM_COLOR
    
    return Image.fromarray(pixels)

def show_matrix(matrix_file, seam_file = None):
    m = np.loadtxt(matrix_file)
    m = np.interp(m, (m.min(), m.max()), (0.0, 1.0))
    m = m * 255
    m = np.stack((m.astype(np.uint8),) * 3, axis=-1)

    if (seam_file is not None):
        seam = np.loadtxt(seam_file).astype(np.uint32)
        for seamPoint in seam:
            m[seamPoint[1]][seamPoint[0]] = SEAM_COLOR

    return Image.fromarray(m)