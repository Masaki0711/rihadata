import dlib
import cv2
import glob
import os
import shutil


if __name__ == "__main__":

    detector = dlib.simple_object_detector('./images/pil/pil.svm')
    files = glob.glob('images/detection/*.jpg')
    target_dir = './images/output'
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    Alli = 0
    fileN=0
    for file in files:
        fileN+=1
        i=0
        print('[Detecting]: '+ file)
        img = cv2.imread(file,0)
        rectangles = detector(img)
        for rect in rectangles:
            x = rect.left()
            y = rect.top()
            w = rect.width()
            h = rect.height()
            if y>0 and x>0:
                i += 1
                cv2.imwrite('./images/output/Bar-'+str(fileN) +'_'+ str(i)+'.jpg', img[y:y+h, x:x+w])
        Alli += i
    print('AllBarN: '+str(Alli))