import pywt
import cv2 # cv2.dct
import numpy as np
import matplotlib.pyplot as plt


image_color = cv2.imread('17.png')

ondelet_type = 'db12'
num_level = 4
delta = [436, 110, 180, 1500, 1]
nbr_bit = [15, 10, 5, 2, 1]
max_tab = np.zeros(6)
size_tab = np.zeros(6)
im_comp_num_bits = np.zeros(6)
r = 0.6


if image_color is not None:
    # image en niveaux de gris
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    im_base_num_bits = int(np.ceil(np.log2(np.max(np.abs(image)) + 1)))
    im_base_size = im_base_num_bits * image.shape[0] * image.shape[1]
    coeffs = pywt.wavedec2(image, ondelet_type, level=num_level)

    c_coded = []
    c = []

    delta[0] = np.ceil((np.max(np.abs(coeffs[0])) + 1) / nbr_bit[0])
    q = np.fix(coeffs[0] / delta[0])
    c_coded.append((q + r * np.sign(q)) * delta[0])

    im_comp_num_bits[0] = int(np.ceil(np.log2(np.max(np.abs(q)) + 1)))
    max_tab[0] = np.max(coeffs[0])
    im_comp_size = im_comp_num_bits[0] * q.size
    size_tab[0] = q.size
    for i in range(1, num_level + 1):
        c_tmp = []
        delta[i] = np.ceil((np.max(np.abs(coeffs[i])) + 1) / nbr_bit[i])
        if (nbr_bit[i] == 0):
            delta[i] = 1000
        for j in range(3):
            q = np.fix(coeffs[i][j] / delta[i])
            c = (q + r * np.sign(q)) * delta[i]
            im_comp_num_bits[i] = int(np.ceil(np.log2(np.max(np.abs(q)) + 1)))
            im_comp_size += im_comp_num_bits[i] * c.shape[0] * c.shape[1]
            c_tmp.append(c)
        max_tab[i] = np.max(coeffs[i])
        size_tab[i] = c.shape[0] * c.shape[1] * 3
        test = tuple(c_tmp)
        c_coded.append(test)
        
    image_comp = pywt.waverec2(c_coded, ondelet_type)
    image_comp = np.clip(image_comp, 0,255)
    ratio = (im_base_size + (num_level + 1) * 32) / im_comp_size

    mse = np.mean((image - image_comp) ** 2)
    rmse = np.sqrt(mse)

    ### Affichage des niveaux d'échelles
    #q_array, _ = pywt.coeffs_to_array(coeffs)
    #plt.imshow(np.log(np.abs(q_array) + 8), 'gray')
    #plt.show()
    
    ### Affichage de l'image compréssée
    cv2.imshow('Image', image_comp.astype(np.uint8))
    cv2.waitKey(0)

    ### Affichage de la différence entre l'image originel et l'image compréssée
    #cv2.imshow('Image', np.abs((image-image_comp)).astype(np.uint8))
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("im8.png", image)
    print(mse)
else:
    print("L'image n'a pas pu être chargée. Vérifiez le chemin d'accès.")