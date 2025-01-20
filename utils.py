import numpy as np

jpeg_quantiz_matrix = np.array([
    [16, 11, 12, 14, 12, 10, 16, 14],
    [13, 14, 18, 17, 16, 19, 24, 40],
    [26, 24, 22, 22, 24, 49, 35, 37],
    [29, 40, 58, 51, 61, 60, 57, 51],
    [56, 55, 64, 72, 92, 78, 64, 68],
    [87, 69, 55, 56, 80, 109, 81, 87],
    [95, 98, 103, 104, 103, 62, 77, 113],
    [121, 112, 100, 120, 92, 101, 103, 99]
])


def chunks(l, n):
    """ Yield successive n-sized chunks from l """
    for i in range(0, len(l), int(n)):
        yield l[i:i + int(n)]


def zig_zag(array, n=None):
    """
    Return a new array where only the first n subelements in zig-zag order are kept.
    The remaining elements are set to 0.
    :param array: 2D array_like
    :param n: Keep up to n subelements. Default: all subelements
    :return: The new reduced array.
    """

    shape = np.array(array).shape

    assert len(shape) >= 2, "Array must be a 2D array_like"

    if n == None:
        n = shape[0] * shape[1]
    assert 0 <= n <= shape[0] * shape[1], 'n must be the number of subelements to return'

    res = np.zeros_like(array)

    (j, i) = (0, 0)
    direction = 'r'  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
    for subel_num in range(1, n + 1):
        res[j][i] = array[j][i]
        if direction == 'r':
            i += 1
            if j == shape[0] - 1:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'dl':
            i -= 1
            j += 1
            if j == shape[0] - 1:
                direction = 'r'
            elif i == 0:
                direction = 'd'
        elif direction == 'd':
            j += 1
            if i == 0:
                direction = 'ur'
            else:
                direction = 'dl'
        elif direction == 'ur':
            i += 1
            j -= 1
            if i == shape[1] - 1:
                direction = 'd'
            elif j == 0:
                direction = 'r'

    return res

