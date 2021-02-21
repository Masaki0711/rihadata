"""
Created on Thu Jan 14 23:21:38 2021

@author: miyashita
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
        # マッチング結果書き出し準備
    # 画像をBGRカラーで読み込み
    base_image_path = './images/detection/bar9.jpg'
    temp_image_path = './bars/bars2.PNG'
    color_base_src = cv2.imread(base_image_path, 1)
    color_temp_src = cv2.imread(temp_image_path, 1)
    # 画像をグレースケールで読み込み0
    gray_base_src = cv2.imread(base_image_path, 0)
    gray_temp_src = cv2.imread(temp_image_path, 0)
    #gray_temp_src= cv2.imread(temp_image_path, 0)
    #tmp = gray_base_src[1100:1350,100:350]
    #gray_temp_src = tmp
    # 特徴点の検出
    t = cv2.AKAZE_create()
    #t = cv2.ORB_create()
    kp_01, des_01 = t.detectAndCompute(gray_base_src, None)
    kp_02, des_02 = t.detectAndCompute(gray_temp_src, None)

    # 画像を読み込む。
    
    # 特徴点を描画する。
    dst = cv2.drawKeypoints(gray_temp_src, kp_01, None)
    plt.gray()
    cv2.namedWindow('w', cv2.WINDOW_NORMAL)
    cv2.imshow("w",dst)
    
    # マッチング処理
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des_01, des_02)
    matches = sorted(matches, key = lambda x:x.distance)
    mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:1], None, flags=2)

    # 結果の表示
    plt.gray()
    plt.figure(figsize=(15,10))
    plt.imshow(mutch_image_src)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #mutch_image_src = cv2.resize(mutch_image_src,(width//2, height//2))
    cv2.imshow("img", mutch_image_src)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
