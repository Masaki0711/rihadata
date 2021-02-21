import dlib

if __name__ == "__main__":

    train_xml = "./images/pil/pil.xml"
    svm_file = "./images/pil/pil.svm"
    options = dlib.simple_object_detector_training_options()
    # 左右対照に学習データを増やすならtrueで訓練(メモリを使う)
    #options.add_left_right_image_flips = True
    # SVMを使ってるのでC値を設定する必要がある
    options.C = 0.001
    
    # スレッド数指定
    options.num_threads = 16
    ## 学習途中の出力をするかどうか
    #options.be_verbose = True
    # 学習許容範囲
    #options.epsilon = 0.001
    # サンプルを増やす最大数(大きすぎるとメモリを使う)
    #options.upsample_limit = 8
    options.detection_window_size=40000
    
    dlib.train_simple_object_detector(train_xml, svm_file, options)
