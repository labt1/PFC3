trtexec --loadEngine=rtdetr_r18_static_int8.trt \
        --avgRuns=1000 \
        --int8 \

#trtexec --loadEngine=def.trt \
#        --shapes=image:1x3x640x640 \
#        --avgRuns=1000 \
#        --fp16