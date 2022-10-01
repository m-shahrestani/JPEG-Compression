import math
import os
import pickle
import numpy as np
import huffman
from PIL import Image
from scipy import fftpack

luminance_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                            [12, 12, 14, 19, 26, 58, 60, 55],
                            [14, 13, 16, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])

chrominance_table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                              [18, 21, 26, 66, 99, 99, 99, 99],
                              [24, 26, 56, 99, 99, 99, 99, 99],
                              [47, 66, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99],
                              [99, 99, 99, 99, 99, 99, 99, 99]])


# part 1
def sampling(image):
    if image.shape[0] % 2 == 0:
        image[1::2, :, 1:3] = image[::2, :, 1:3]
    else:
        last_row = image.shape[0] - 2
        image[1::2, :, 1:3] = image[:last_row:2, :, 1:3]
    if image.shape[1] % 2 == 0:
        image[:, 1::2, 1:3] = image[:, ::2, 1:3]
    else:
        last_col = image.shape[1] - 2
        image[:, 1::2, 1:3] = image[:, :last_col:2, 1:3]
    return image


# part 2
def convert_dc_ac(image, width=8):
    if image.shape[0] % 4 != 0:
        short = math.ceil(image.shape[0] / 4) * 4 - image.shape[0]
        image = np.append(image, np.zeros(shape=(short, image.shape[1], 3)), axis=0)
    if image.shape[1] % 4 != 0:
        short = math.ceil(image.shape[1] / 4) * 4 - image.shape[1]
        image = np.append(image, np.zeros(shape=(image.shape[0], short, 3)), axis=1)
    blocks_count = (image.shape[0] // width) * (image.shape[1] // width)
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, (width * width - 1), 3), dtype=np.int32)
    block_index = 0
    num_elements = width * width
    for i in range(0, image.shape[0], width):
        for j in range(0, image.shape[1], width):
            block = image[i:i + width, j:j + width, :]
            dct_block = dct2d(block)
            q_block = quantization(dct_block)
            z_block = zigzag(q_block)
            dc[block_index, :] = z_block[0, :]
            ac[block_index, :, :] = z_block[1:num_elements, :]
            block_index += 1
    return dc, ac


def dct2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


# part 3
def quantization(dctb):
    dctb[:, :, 0] = dctb[:, :, 0] / luminance_table
    dctb[:, :, 1] = dctb[:, :, 1] / chrominance_table
    dctb[:, :, 2] = dctb[:, :, 2] / chrominance_table
    return dctb


# part 5
def zigzag(block):
    zz = np.empty(shape=(block.shape[0] * block.shape[1], block.shape[2]))
    for n in range(3):
        zz[:, n] = np.concatenate([np.diagonal(block[::-1, :, n], k)[::(2 * (k % 2) - 1)] for k in range(1 - block.shape[0], block.shape[0])])[:]
    return zz


# part 6
def compress_jpeg(dc, ac, dc_y, dc_c, ac_y, ac_c):
    with open('compressed_jpeg.bin', 'wb') as file:
        for i in range(ac.shape[0]):
            for j in range(3):
                if j == 0:
                    dc_table, ac_table = dc_y, ac_y
                else:
                    dc_table, ac_table = dc_c, ac_c
                file.write(dc_table[dc[i, j]])
                sym = huffman.RLC(ac[i, :, j])
                for m in range(len(sym)):
                    file.write(ac_table[tuple(sym[m])])
        pickle.dump(dc_y, file)
        pickle.dump(dc_c, file)
        pickle.dump(ac_y, file)
        pickle.dump(ac_c, file)


def print_size(path):
    size = os.path.getsize(path)
    print('Size of', path, 'is', size, 'bytes')


if __name__ == '__main__':
    # part 1
    img = Image.open('photo1.png')
    ycbcr = img.convert('YCbCr')
    chroma_subsampling = sampling(np.array(ycbcr, dtype=np.uint8))
    # part 2 & 3
    dc, ac = convert_dc_ac(chroma_subsampling)
    # part 4
    # Q3
    array = [8, 8, 34, 5, 10, 34, 6, 43, 127, 10, 10, 8, 10, 34, 10]
    tree = huffman.HuffmanTree(array)
    table = tree.get_value_to_bitstring_table()
    bits = bytes()
    for i in array:
        bits += (table[i])
    print('output is', bits)
    print('rate of compression is', (len(array)*8)/len(bits))
    # part 5 & 6
    dc_y, dc_c, ac_y, ac_c = huffman.hoffman_coding(dc, ac)
    # Q4
    compress_jpeg(dc, ac, dc_y, dc_c, ac_y, ac_c)
    print_size('photo1.png')
    print_size('compressed_jpeg.bin')
