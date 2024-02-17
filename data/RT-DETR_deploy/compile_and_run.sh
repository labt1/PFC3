# rm -rf build && mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
mkdir -p build && cd build && cmake ..  && make -j48 && cd ..
cd output

# Para una imagen (test) -c 100 -w 10 -b 1
#: '
./infer -f ../models/rtdetr_r18_static_new_fp16.trt \
        -i ../input/1.jpg \
        -o result \
        -c 1 -w 1 -b 1 -s 0.30
#'


: '
./infer -f ../models/rtdetr_r18_static_fp16.trt \
        -s 0.50f -b 1\
        -v "../input/test2.mp4"
'


: '
./infer -f ../models/rtdetr_r18_static_fp16.trt \
        -s 0.75f -b 1\
        -p "rtsp://admin:unsa2024@192.168.0.216:554/cam/realmonitor?channel=1&subtype=0"
'


cd ..