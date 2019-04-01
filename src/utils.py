import numpy as np
import os


def add_offset(w, h, bbox, offset):

    crop_h = int(h * (float(bbox[1]) - float(bbox[0])))
    crop_w = int(w * (float(bbox[3]) - float(bbox[2])))

    new_w = crop_w / (1 - float(offset[2]) - float(offset[3]) + 1e-10)
    new_h = crop_h / (1 - float(offset[0]) - float(offset[1]) + 1e-10)

    r_w = min(w, max(0, new_w))
    r_h = min(h, max(0, new_h))

    x1 = max(0, h * float(bbox[0]) - r_h * float(offset[0]))
    x2 = min(h, x1 + r_h)
    y1 = max(0, w * float(bbox[2]) - r_w * float(offset[2]))
    y2 = min(w, y1 + r_w)

    bbox_aes = [x1 / float(h), x2 / float(h), y1 / float(w), y2 / float(w)]
    #print crop_h, crop_w, new_w, new_h, bbox, offset, bbox_aes

    return bbox_aes


def recover_from_normalization_with_order(w, h, bbox):
    #print w, h, bbox
    box = [max(0, int(bbox[2] * w)), max(0, int(bbox[0] * h)), min(w, int(bbox[3] * w)), min(h, int(bbox[1] * h))]
    return box


def recover_from_normalization(w, h, bbox):
    #print w, h, bbox
    box = [max(0, int(bbox[0] * h)), min(h, int(bbox[1] * h)), max(0, int(bbox[2] * w)), min(w, int(bbox[3] * w))]
    return box


def normalization(w, h, bbox):
    box = [min(max(0.0, float(bbox[0]) / h), 1.0),
           min(max(0.0, float(bbox[1]) / h), 1.0),
           min(max(0.0, float(bbox[2]) / w), 1.0),
           min(max(0.0, float(bbox[3]) / w), 1.0)]
    return box


def get_association_matrix(G):
    assert len(G.shape) == 2, 'Dim of Matrix G should be 2, got %d' % len(G.shape)
    m = G.shape[0]
    n = G.shape[1]

    G1 = np.empty(G.shape, dtype=np.float32)
    G1[0][0] = G[0][0]
    for i in xrange(m):

        for j in xrange(n):
            if i == 0:
                a = 0
            else:
                a = G1[i - 1][j]
            if j == 0:
                b = 0
            else:
                b = G1[i][j - 1]
            if i != 0 and j != 0:
                c = G1[i - 1][j - 1]
            else:
                c = 0
            G1[i][j] = G[i][j] + a + b - c

    Gc = np.empty(G.shape, dtype=np.float32)
    Gc[0][0] = G[0][0]

    for i in xrange(m):
        for j in xrange(n):
            if i == 0:
                a = 0
            else:
                a = Gc[i - 1][j]
            Gc[i][j] = G[i][j] + a
    return G1, Gc


def Minimum_Rectangle(G, r):

    m = G.shape[0]
    n = G.shape[1]
    #print m, n
    G1, Gc = get_association_matrix(G)
    i = 0
    j = 0
    w = float(9999999999)
    h = float(9999999999)
    Smin = -1

    T = G1[m - 1][n - 1] * r

    for i1 in xrange(1, m):
        for i2 in xrange(i1, m):
            a = (Gc[i2, :] - Gc[i1 - 1, :])
            j1, j2, S0 = shortestSubarray(a, T)
            if j1 > 0 and j2 > 0:
                w0 = j2 - j1 + 1
                h0 = i2 - i1 + 1
                if w0 * h0 < w * h or (w0 * h0 == w * h and S0 > Smin):
                    i = i1
                    j = j1
                    w = w0
                    h = h0
                    Smin = S0
    return i, j, w, h


def shortestSubarray(a, T):
    j1 = 0
    j2 = 0
    Lmin = float(999999999)
    Smin = -1
    st = 0
    ed = 0
    S0 = a[0]
    n = len(a)
    if T > 0:
        while st < n:
            if S0 < T:
                ed += 1
                if ed >= n:
                    break
                else:
                    S0 += a[ed]
            else:
                L = ed - st + 1
                if L < Lmin or (L == Lmin and S0 > Smin):
                    j1 = st
                    j2 = ed
                    Lmin = L
                    Smin = S0
                S0 = S0 - a[st]
                st += 1
    return j1, j2, Smin