import numpy as np

luminance_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

chrominance_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

def chunks(l, n):
    """ Yield successive n-sized chunks from l """
    for i in range(0, len(l), int(n)):
        yield l[i:i + int(n)]


def zig_zag(array, n=None):
    """
    Return a new array where only the first n elements in zig-zag order are kept.
    The remaining elements are set to 0.
    :param array: 2D array_like
    :param n: Keep up to n elements. Default: all elements
    :return: The new reduced array.
    """

    shape = np.array(array).shape

    assert len(shape) >= 2, "Array must be 2D"

    if n == None:
        n = shape[0] * shape[1]
    assert 0 <= n <= shape[0] * shape[1], 'Number of elements to keep must be between 0 and the total number of elements'

    res = np.zeros_like(array)

    (r, c) = (0, 0)
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    for _ in range(0, n):
        res[r][c] = array[r][c]
        if direction == 'r':
            c += 1
            if r == shape[0] - 1:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            c -= 1
            r += 1
            if r == shape[0] - 1:
                direction = 'r'
            elif c == 0:
                direction = 'd'
        elif direction == 'd':
            r += 1
            if c == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            c += 1
            r -= 1
            if c == shape[1] - 1:
                direction = 'd'
            elif r == 0:
                direction = 'r'

    return res

