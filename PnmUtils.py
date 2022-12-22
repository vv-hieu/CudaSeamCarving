from PIL import Image
import numpy as np

SEAM_COLOR = (255, 0, 255)

def show_pnm(file_name, seam = None):
    try:
        with open(file_name, 'r') as f:
            string = f.read().replace('\n', ' ').replace('\r', ' ')
            vals = string.split()
            w = int(vals[1])
            h = int(vals[2])
            if vals[0] == 'P1':
                pixels = (np.array(vals[3:], dtype=np.uint8).reshape((h, w)) * 255).astype(np.uint8)
            elif vals[0] == 'P2':
                maxVal = int(vals[3])
                pixels = (np.array(vals[4:], dtype=np.uint8).reshape((h, w)) / maxVal * 255).astype(np.uint8)
            elif vals[0] == 'P3':
                maxVal = int(vals[3])
                pixels = (np.array(vals[4:], dtype=np.uint8).reshape((h, w, 3)) / maxVal * 255).astype(np.uint8)
    except:
        with open(file_name, 'rb') as f:
            binary = f.read()
            string = str(binary).replace('\\n', ' ').replace('\\r', ' ').strip("b'").strip("'")
            vals = string.split()
            w = int(vals[1])
            h = int(vals[2])
            if vals[0] == 'P4': 
                string2 = string.lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[0]).lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[1]).lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[2]).lstrip(' ').lstrip('\n').lstrip(' ')
                l = len(string) - len(string2)
                pixels = (np.unpackbits(np.frombuffer(binary[l:], dtype=np.uint8)).reshape((h, w))).astype(np.uint8) * 255
            elif vals[0] == 'P5':
                maxVal = int(vals[3])
                string2 = string.lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[0]).lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[1]).lstrip(' ').lstrip(vals[2]).lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[3]).lstrip(' ').lstrip('\n').lstrip(' ')
                l = len(string) - len(string2)
                pixels = (np.frombuffer(binary[l:], dtype=np.uint8).reshape((h, w)) / maxVal * 255).astype(np.uint8)
            elif vals[0] == 'P6':
                maxVal = int(vals[3])
                string2 = string.lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[0]).lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[1]).lstrip(' ').lstrip(vals[2]).lstrip(' ').lstrip('\n').lstrip(' ').lstrip(vals[3]).lstrip(' ').lstrip('\n').lstrip(' ')
                l = len(string) - len(string2)
                pixels = (np.frombuffer(binary[l:], dtype=np.uint8).reshape((h, w, 3)) / maxVal * 255).astype(np.uint8)
            else:
                return

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
    