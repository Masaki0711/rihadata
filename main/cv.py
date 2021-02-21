# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:21:38 2021

@author: miyashita
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import gaussian_kde
from scipy import signal


if __name__ == '__main__':
    #線分のposition
    pos = [[190, 112],
             [225, 372],
             [227, 670],
             [144, 1003],
             [144, 1169],
             [130, 1403],
             [546, 111],
             [443, 333],
             [527, 604],
             [447, 899],
             [436, 1127],
             [431, 1334],
             [782, 268],
             [755, 639],
             [790, 820],
             [778, 1033],
             [782, 1253],
             [810, 1449],
             [1381, 91],
             [1380, 331],
             [1359, 515],
             [1395, 797],
             [1448, 1101],
             [1379, 1379],
             [1699, 309],
             [1718, 507],
             [1683, 709],
             [1699, 943],
             [1680, 1223],
             [1640, 1502],
             [2020, 156],
             [2024, 423],
             [2089, 629],
             [2017, 981],
             [2045, 1179],
             [2054, 1403]]
    def hough(img,j):
        #前処理
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
        #lines = cv2.HoughLines(edges,1,np.pi/180,200)
        lines = cv2.HoughLines(edges,1,np.pi/180,10)
        #for rho,theta in lines[0]:
        thetas=lines[:,0,1]
        
        #カーネル密度推定
        kde = gaussian_kde(thetas)
        # 0〜31で密度推定
        xticks = np.linspace(0, max(thetas))
        estimates = kde(xticks)
        probs = kde.evaluate(xticks)
        #plt.plot(estimates)
        maxId = signal.argrelmax(estimates)
        topsID=[]
        es=estimates[maxId][estimates[maxId]>(max(estimates[maxId])*0.05)]#極大値の最大値指定
        for e in es:
            topsID.append(maxId[0][list(estimates[maxId]).index(e)])
        maxId=topsID
        print(j,estimates[maxId],max(maxId)-min(maxId)>=18)
        if len(maxId)>=2 and max(maxId)-min(maxId)>=18:　#極大値の幅によって判定
            cv2.putText(color_src, str(j) + ") checked",(pos[j-1][0],pos[j-1][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),thickness=2)
            
        else:
            cv2.putText(color_src, str(j) + ") unchecked",(pos[j-1][0],pos[j-1][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),thickness=2)
        
        
        """図示
        plt.plot(xticks[maxId], estimates[maxId], "ro")
        plt.plot(xticks, probs)
        plt.show()"""
        """for rho,theta in lines[10]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
    
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)"""
    
        #print('number of lines: ', len(lines))
        #print(pd.Series(thetas, name='values').mode())
        #print(lines)
        """plt.figure(figsize=(4, 3))
        #plt.hist(thetas,range=(0,np.pi), rwidth=0.9)
        plt.hist(thetas,range=(0,np.pi), rwidth=0.9,bins=180)
        plt.xlabel('theta')
        plt.ylabel('point')
        plt.show()
        thetas=sorted(list(set(thetas)))
        #[print(x) for x in thetas]
        # show results
        plt.figure(figsize=(5, 5))
        plt.imshow(gray, cmap=cm.gray)
        plt.subplot(2, 2, 2)
        plt.imshow(edges)
        plt.subplot(2, 2, 3)
        plt.imshow(img)
        #plt.subplot(2, 2, 4)
        #plt.imshow(out, cmap=cm.gray)"""

    target = 1 # 対象の試験データ
    thr = 0.854 #二値化の閾値
    
    
    
    for i in range(target,target+10):
        base_image_path = './images/detection/bar'+str(i)+'.jpg'
        gray_base_src = cv2.imread(base_image_path, 0)
        # 閾値の設定
        threshold = 245
        # 二値化(閾値100を超えた画素を255にする。)
        ret, gray_base_src = cv2.threshold(gray_base_src, threshold, 255, cv2.THRESH_BINARY)
        gray_base_src= cv2.medianBlur(gray_base_src, ksize=3)
        color_src = cv2.cvtColor(gray_base_src, cv2.COLOR_GRAY2BGR)
        cnt=0
        for j in range(1,37):
            # 対象画像を指定
            temp_image_path = './bars/bars'+str(j)+'.PNG'
            #temp_image_path = './bar2.PNG'
            # 画像をグレースケールで読み込み
            gray_temp_src = cv2.imread(temp_image_path, 0)
            ret, gray_temp_src = cv2.threshold(gray_temp_src, threshold, 255, cv2.THRESH_BINARY)
            gray_temp_src=cv2.medianBlur(gray_temp_src, ksize=3)
        
            # ラベリング処理
            label = cv2.connectedComponentsWithStats(gray_base_src)
            n = label[0] - 1
            data = np.delete(label[2], 0, 0)
        
            # マッチング結果書き出し準備
            height, width = gray_temp_src.shape[:2]
        
            res = cv2.matchTemplate(gray_base_src, gray_temp_src, cv2.TM_CCOEFF_NORMED, height*width)
            res_n = cv2.minMaxLoc(res)
            res_num = res_n[1]
            #cv2.rectangle(color_src, (x0, y0), (x1, y1), (0, 0, 255))"""
            #hough(color_src[pos[j-1][1]:pos[j-1][1]+height,pos[j-1][0]:pos[j-1][0]+width],j)
            r=res_n[3][0]//305
            if r >= 4:
                r -= 1
            if (j-1)//6 == r:
                if res_num >= thr:
                        cnt+=1
                        #cv2.circle(color_src, (res_n[3][0],res_n[3][1]), 10, (0, 0, 255), thickness=2)
                        print('Ditect! Bar'+str(i)+'-'+str(j)+' sim:'+str(round(res_num, 3)))
                        #cv2.rectangle(gray_base_src, (res_n[3][0], res_n[3][1]),(res_n[3][0]+width, res_n[3][1]+height), (0, 0, 0))
                        #cv2.putText(color_src, str(j) + ") ",(res_n[3][0],res_n[3][1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),thickness=2)
                        cv2.putText(color_src, str(j) + ") " +str(round(res_num, 4)),(res_n[3][0],res_n[3][1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),thickness=2)
                        cv2.putText(gray_base_src, str(j) + ") " +str(round(res_num, 4)),(res_n[3][0],res_n[3][1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255),thickness=2)
                elif res_num < thr:
                        #cv2.circle(color_src, (res_n[3][0],res_n[3][1]), 10, (0, 0, 255), thickness=2)
                        #print('Ditect! Bar'+str(i)+'-'+str(j)+' sim:'+str(round(res_num, 3)))
                        #cv2.rectangle(gray_base_src, (res_n[3][0], res_n[3][1]),(res_n[3][0]+width, res_n[3][1]+height), (0, 0, 0))
                        #cv2.putText(color_src, str(j) + ") " ,(res_n[3][0],res_n[3][1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),thickness=2)
                        cv2.putText(color_src, str(j) + ") " +str(round(res_num, 4)),(res_n[3][0],res_n[3][1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),thickness=2)
                        cv2.putText(gray_base_src, str(j) + ") " +str(round(res_num, 4)),(res_n[3][0],res_n[3][1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),thickness=2)
                              
            #絶対値の求めたのち、背景差分を求める
            #gray_base_src[res_n[3][0]:res_n[3][0]+height,res_n[3][1]:res_n[3][1]+width]
            gray_base_src[res_n[3][1]:res_n[3][1]+height,res_n[3][0]:res_n[3][0]+width] = cv2.absdiff(gray_base_src[res_n[3][1]:res_n[3][1]+height,res_n[3][0]:res_n[3][0]+width]
            ,gray_temp_src)
        
            
            
        # 結果の表示
        #plt.gray()
        """plt.figure(figsize=(15,10))
        plt.imshow(color_src)
        cv2.namedWindow('color_src', cv2.WINDOW_NORMAL)
        cv2.imshow("color_src", color_src)"""
        cv2.putText(color_src, "score: " +str(36-cnt),(50,1600), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),thickness=2)
        #cv2.imwrite('./outputHough/hough-'+str(i)+'.jpg',color_src)
        #cv2.imwrite('./output/Scored-'+str(i)+'.jpg',color_src)
        cv2.imwrite('./output3/Scored-'+str(i)+'.jpg',gray_base_src)
        #print('score:',36-cnt)
    
    
    
    """
    #点を表示する (x, y) 座標を作成
    X, Y = np.indices(res.shape)
    
    # 描画する。
    fig = plt.figure(figsize=(10, 10))
    axes = fig.add_subplot(111, projection="3d")
    points = axes.scatter(
        X.flat,
        Y.flat,
        res.flat,
        c=res.flat,
        cmap="jet",
        #edgecolor="gray",
        s=50,
    )
    axes.plot_surface(X, Y, res, cmap="jet", alpha=0.1)
    
    # カラーバー追加 (xmin, ymin, w, h) でカラーバーを表示する位置を指定
    cbar_ax = fig.add_axes((0.9, 0.3, 0.02, 0.4))
    cbar = fig.colorbar(points, cax=cbar_ax)
    #cbar.set_label("")
    
    # 軸ラベル設定
    axes.set_xlabel("height")
    axes.set_ylabel("width")
    
    # Title表示
    axes.set_title("similarity", fontsize=16)
    
    # 余白調整
    plt.subplots_adjust(right=0.85)
    plt.subplots_adjust(wspace=0.15)
    
    # 視点
    axes.view_init(60, -10)"""

    
    
    
    
    
    
    
    
    
    """# 特徴点マッチング
    # 画像をBGRカラーで読み込み
    color_base_src = cv2.imread(base_image_path, 1)
    color_temp_src = cv2.imread(temp_image_path, 1)
    # 画像をグレースケールで読み込み
    gray_base_src = cv2.imread(base_image_path, 0)
    #gray_temp_src= cv2.imread(temp_image_path, 0)
    #tmp = gray_base_src[1100:1350,100:350]
    gray_temp_src = tmp
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
    mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:40], None, flags=2)

    # 結果の表示
    plt.gray()
    plt.figure(figsize=(15,10))
    plt.imshow(mutch_image_src)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #mutch_image_src = cv2.resize(mutch_image_src,(width//2, height//2))
    cv2.imshow("img", mutch_image_src)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
    
    
    

    