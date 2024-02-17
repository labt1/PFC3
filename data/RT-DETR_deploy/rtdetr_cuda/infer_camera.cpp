#include "common/arg_parsing.hpp"
#include "common/cv_cpp_utils.hpp"
#include "rtdetr_detect.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "camera/RTSPcam.h"

int trt_cuda_video_stream_inference(ai::arg_parsing::Settings *s)
{
    tensorrt_infer::rtdetr_cuda::RTDETRDetect rtdetr_obj;
    rtdetr_obj.initParameters(s->model_path, s->score_thr);
    ai::cvUtil::Image input;
    ai::cvUtil::BoxArray output;
    ai::utils::Timer timer;

    float latency;
    float avg_latency = 0;
    int frames_cont = 0;

    RTSPcam cam;
    cam.Open(s->camera_ip);
    cv::Mat frame;
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    while(true) {
        
        if(!cam.GetLatestFrame(frame)){
            cout << "Capture read error" << endl;
            break;
        }
        
        timer.start();

        input = ai::cvUtil::cvimg_trans_func(frame);
        output = rtdetr_obj.forward(input);
        cv::putText(frame, "FPS: " + std::to_string(1000/latency), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        ai::cvUtil::draw_one_image_rectangle(frame, output, s->classlabels);

        latency = timer.stop("Timer", 1, false);
        avg_latency += latency;
        frames_cont++;

        cv::imshow("Camera",frame);
        char esc = cv::waitKey(1);
        if(esc == 27) break;
    }

    std::cout<<"Promedio latencia: "<<avg_latency/frames_cont<<std::endl;
    std::cout<<"Promedio FPS: "<<1000/(avg_latency/frames_cont)<<std::endl;

    std::cout<<"Promedio preprocess: "<<rtdetr_obj.preprocess_time/frames_cont<<std::endl;
    std::cout<<"Promedio inference: "<<rtdetr_obj.inference_time/frames_cont<<std::endl;
    std::cout<<"Promedio postprocess: "<<rtdetr_obj.postprocess_time/frames_cont<<std::endl;

    cv::destroyAllWindows();
}