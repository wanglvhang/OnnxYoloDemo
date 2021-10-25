using Lvhang.WindowsCapture;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    public class YOLOv4Detector: IYOLODetector
    {

        private InferenceSession _onnxSession;
        private Mutex _sessionMutex = new Mutex();

        public YOLOv4Detector(string model_path)
        {
            // Session Options
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            //options.EnableProfiling = true;
            //options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            //options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            //options.OptimizedModelFilePath = ".\\Models\\optimized\\opt_yolov4.onnx";
            //options.AppendExecutionProvider_OpenVINO(@"MYRIAD_FP16");
            options.AppendExecutionProvider_DML(0);
            //options.AppendExecutionProvider_CPU(0);

            // create inference session
            _onnxSession = new InferenceSession(model_path, options);

            //var providers = OrtEnv.Instance().GetAvailableProviders();

        }


        public ProcessDetail Inference(Bitmap bitmapOrg)
        {

            var sw = new Stopwatch();
            sw.Start();
            var new_image = bitmapOrg.ResizeWithoutPadding(416, 416);
            sw.Stop();
            var pd = new ProcessDetail(Thread.CurrentThread.ManagedThreadId, sw.ElapsedMilliseconds, 0, 0, 0);


            sw.Reset(); sw.Start();
            var input_tensor = new_image.FastToOnnxTensor_13hw();
            sw.Stop();
            pd = pd with { ToTensorCost = sw.ElapsedMilliseconds };



            sw.Reset(); sw.Start();
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("input_1:0", input_tensor));
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            sw.Stop();
            pd = pd with { InferenceCost = sw.ElapsedMilliseconds };



            sw.Reset(); sw.Start();
            var resultsArray = results.ToArray();
            Tensor<float> tensors = resultsArray[0].AsTensor<float>();
            var detectioinResults = this.parseYoloPredictions(tensors.ToArray());
            bitmapOrg.DrawYoloPrediction(detectioinResults, Brushes.Green, Brushes.White);
            sw.Stop();
            pd = pd with { DrawCost = sw.ElapsedMilliseconds };

            return pd;

        }


        public Task<ProcessDetail> InferenceAsync(Bitmap bitmapOrg)
        {
            return Task.Run(() =>
            {
                lock (this._onnxSession)
                {

                    var ptime = Inference(bitmapOrg);
                    return ptime;

                }

            });
        }


        public void Stop()
        {
            _onnxSession?.EndProfiling();
            _onnxSession?.Dispose();
        }


        private List<YoloPrediction> parseYoloPredictions(float[] results)
        {
            int c_values = 84;
            int c_boxes = results.Length / c_values;
            float confidence_threshold = 0.5f;
            List<YoloPrediction> detections = new List<YoloPrediction>();
            for (int i_box = 0; i_box < c_boxes; i_box++)
            {
                float max_prob = 0.0f;
                int label_index = -1;
                for (int j_confidence = 4; j_confidence < c_values; j_confidence++)
                {
                    int index = i_box * c_values + j_confidence;
                    if (results[index] > max_prob)
                    {
                        max_prob = results[index];
                        label_index = j_confidence - 4;
                    }
                }
                if (max_prob > confidence_threshold)
                {

                    var bbox = new BBox(results[i_box * c_values + 1], results[i_box * c_values + 0], results[i_box * c_values + 3], results[i_box * c_values + 2]);

                    detections.Add(new YoloPrediction()
                    {
                        Box = bbox,
                        Confidence = max_prob,
                        LabelIndex = label_index,
                        LabelName = YoloPrediction.YoloLabes[label_index]
                    });
                }
            }

            // Non-maximum Suppression(NMS), a technique which filters the proposals 
            // based on Intersection over Union(IOU)
            return YoloPrediction.NMS(detections);
        }


    }




    //    private static readonly string[] _labels =
    //     {
    //            "person",
    //            "bicycle",
    //            "car",
    //            "motorbike",
    //            "aeroplane",
    //            "bus",
    //            "train",
    //            "truck",
    //            "boat",
    //            "traffic light",
    //            "fire hydrant",
    //            "stop sign",
    //            "parking meter",
    //            "bench",
    //            "bird",
    //            "cat",
    //            "dog",
    //            "horse",
    //            "sheep",
    //            "cow",
    //            "elephant",
    //            "bear",
    //            "zebra",
    //            "giraffe",
    //            "backpack",
    //            "umbrella",
    //            "handbag",
    //            "tie",
    //            "suitcase",
    //            "frisbee",
    //            "skis",
    //            "snowboard",
    //            "sports ball",
    //            "kite",
    //            "baseball bat",
    //            "baseball glove",
    //            "skateboard",
    //            "surfboard",
    //            "tennis racket",
    //            "bottle",
    //            "wine glass",
    //            "cup",
    //            "fork",
    //            "knife",
    //            "spoon",
    //            "bowl",
    //            "banana",
    //            "apple",
    //            "sandwich",
    //            "orange",
    //            "broccoli",
    //            "carrot",
    //            "hot dog",
    //            "pizza",
    //            "donut",
    //            "cake",
    //            "chair",
    //            "sofa",
    //            "pottedplant",
    //            "bed",
    //            "diningtable",
    //            "toilet",
    //            "tvmonitor",
    //            "laptop",
    //            "mouse",
    //            "remote",
    //            "keyboard",
    //            "cell phone",
    //            "microwave",
    //            "oven",
    //            "toaster",
    //            "sink",
    //            "refrigerator",
    //            "book",
    //            "clock",
    //            "vase",
    //            "scissors",
    //            "teddy bear",
    //            "hair drier",
    //            "toothbrush"
    //    };

}
