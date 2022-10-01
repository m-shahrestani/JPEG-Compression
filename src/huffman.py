from queue import PriorityQueue


# part 4
# https://github.com/ghallak/jpeg-python

def hoffman_coding(dc, ac):
    bc = ac.shape[0]
    huffman_dc_y = HuffmanTree(DPCM(dc[:, 0]))
    huffman_dc_c = HuffmanTree(DPCM(dc[:, 1:].flat))
    huffman_ac_y = HuffmanTree(flatten(RLC(ac[i, :, 0]) for i in range(bc)))
    huffman_ac_c = HuffmanTree(flatten(RLC(ac[i, :, j]) for i in range(bc) for j in [1, 2]))
    return huffman_dc_y.get_value_to_bitstring_table(), \
           huffman_dc_c.get_value_to_bitstring_table(), \
           huffman_ac_y.get_value_to_bitstring_table(), \
           huffman_ac_c.get_value_to_bitstring_table()


class _Node:
    def __init__(self, value=None, freq=None, left_child=None, right_child=None):
        self.value = value
        self.freq = freq
        self.left_child = left_child
        self.right_child = right_child

    def init_leaf(self, value, freq):
        return _Node(value, freq, None, None)

    def init_node(self, left_child, right_child):
        freq = left_child.freq + right_child.freq
        return _Node(None, freq, left_child, right_child)

    def is_leaf(self):
        return self.value is not None

    def __eq__(self, other):
        if self.value == other.value and \
                self.freq == other.freq and \
                self.left_child == other.left_child and \
                self.right_child == other.right_child:
            return True
        return False

    def __nq__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.freq < other.freq

    def __le__(self, other):
        return self.freq < other.freq or self.freq == other.freq

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)


class HuffmanTree:
    def __init__(self, arr):
        self.value_to_bitstring_table = dict()
        q = PriorityQueue()

        for val, freq in cal_freq(arr).items():
            node = _Node().init_leaf(value=val, freq=freq)
            q.put(node)

        while q.qsize() >= 2:
            left = q.get()
            right = q.get()
            node = _Node().init_node(left_child=left, right_child=right)
            q.put(node)

        self.root = q.get()
        self.create_huffman_table()

    def create_huffman_table(self):
        def tree_traverse(current_node, bits=''):
            if current_node is None:
                return
            if current_node.is_leaf():
                self.value_to_bitstring_table[current_node.value] = bits.encode()
                return
            tree_traverse(current_node.left_child, bits + '0')
            tree_traverse(current_node.right_child, bits + '1')

        tree_traverse(self.root)

    def get_value_to_bitstring_table(self):
        return self.value_to_bitstring_table


def cal_freq(arr):
    freq = dict()
    for i in arr:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1
    return freq


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''
    binstr = bin(abs(n))[2:]
    return binstr if n > 0 else binstr_flip(binstr)


def binstr_flip(binstr):
    if not set(binstr).issubset('01'):
        raise ValueError("binstr should have only '0's and '1's")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def DPCM(list):
    list[1:] = list[1:] - list[0]
    return list


def RLC(list):
    last_nonzero = -1
    for i, s in enumerate(list):
        if s != 0:
            last_nonzero = i
    symbols = []
    run_length = 0
    for i, s in enumerate(list):
        if i > last_nonzero:
            symbols.append((0, 0))
            break
        elif s == 0 and run_length < 15:
            run_length += 1
        else:
            symbols.append((run_length, s))
            run_length = 0
    return symbols
