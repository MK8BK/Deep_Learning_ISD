import numpy as np


def convolution(img:  np.ndarray, detector: np.ndarray) -> np.ndarray:
    """
        Convolution d'une image par une matrice detectrice
        @param: <img>:np.ndarray ; l'image au format numpy
        @param: <detector>:np.ndarray ; la matrice detectrice au format numpy
        @return: <img_out>:np.ndarray ; la convolution de img par detector
    """
    assert(detector.shape[0]%2==1 and detector.shape[1]%2==1),\
        f"Filtre de taille paire: {detector.shape[0]}x{detector.shape[1]}"
    n, m = img.shape
    d = detector.shape
    h_conv = np.floor(n + - d[0] ).astype(int) + 1
    w_conv = np.floor(m + - d[1] ).astype(int) + 1
    img_out = np.zeros((h_conv, w_conv))
    b = d[0] // 2, d[1] // 2
    center_w = b[0] * 1
    center_h = b[1] * 1
    
    for i in range(h_conv):
        center_x = center_w + i * 1
        indices_x = [center_x + l * 1 for l in range(-b[0], b[0] + 1)]
        for j in range(w_conv):
            center_y = center_h + j * 1
            indices_y = [center_y + l * 1 for l in range(-b[1], b[1] + 1)]
            subimg = img[indices_x, :][:, indices_y]
            img_out[i][j] = np.sum(np.multiply(subimg, detector))
    return img_out


def pool(img: np.ndarray, pool_shape: Tuple[int,int]=(2,2),
                         pool_type: str="max")->np.ndarray:
    """
        poo
    """
    h, w = pool_shape
    #assert(img.shape[0]%h==0 and img.shape[1]%w==0)
    nimg = []
    for i in range(0,img.shape[0],h):
        nimg.append([])
        for j in range(0,img.shape[1],w):
            if pool_type=="max":
                val = np.max(img[i:i+h,j:j+w])
            elif pool_type=="average":
                if(img.dtype == np.dtype('int32')):
                    val = int(np.average(img[i:i+h,j:j+w]))
                else:
                    val = np.mean(img[i:i+h,j:j+w])
            nimg[-1].append(val)
    return np.array(nimg)

