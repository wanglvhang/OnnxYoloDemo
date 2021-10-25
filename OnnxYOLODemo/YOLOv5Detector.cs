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
    public class YOLOv5Detector : IYOLODetector
    {

        private InferenceSession _onnxSession;
        private Mutex _sessionMutex = new Mutex();

        public YOLOv5Detector(string model_path, bool useDirectML)
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            //options.EnableProfiling = true;
            //options.ProfileOutputPathPrefix = "yolov3_profile";

            if (useDirectML)
            {
                options.AppendExecutionProvider_DML(0);
            }
            else
            {
                options.IntraOpNumThreads = 2;
                options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                options.InterOpNumThreads = 6;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.OptimizedModelFilePath = ".\\Models\\optimized\\opt_yolov5s.onnx";
                options.AppendExecutionProvider_CPU(0);
            }

            // create inference session
            _onnxSession = new InferenceSession(model_path, options);
        }

        public ProcessDetail Inference(Bitmap img)
        {

            //var ptime = new ProcessDetail();

            var sw = new Stopwatch();
            sw.Start();
            var resized_image = img.Resize(640, 640);
            sw.Stop();
            var pd = new ProcessDetail(Thread.CurrentThread.ManagedThreadId, sw.ElapsedMilliseconds, 0, 0, 0);


            sw.Reset(); sw.Start();
            var input_tensor = resized_image.FastToOnnxTensor_13hw();
            sw.Stop();
            pd = pd with { ToTensorCost = sw.ElapsedMilliseconds };


            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("images", input_tensor));


            sw.Reset(); sw.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            sw.Stop();
            pd = pd with { InferenceCost = sw.ElapsedMilliseconds };


            sw.Reset(); sw.Start();
            var resultsArray = results.ToArray();
            Tensor<float> tensors = resultsArray[0].AsTensor<float>();


            var array = tensors.ToArray();
            var preds = this.ParseResults(array);

            img.DrawYoloPrediction(preds, Brushes.YellowGreen, Brushes.YellowGreen);

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
            //_onnxSession?.EndProfiling();
            _onnxSession?.Dispose();
        }


        private List<YoloPrediction> ParseResults(float[] results)
        {
            //output  1 25200 85
            //box format  0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence
            int dimensions = 85;
            int rows = results.Length / dimensions;
            int confidenceIndex = 4;
            int labelStartIndex = 5;

            float inputlWidth = 640;
            float inputHeight = 640;


            List<YoloPrediction> detections = new List<YoloPrediction>();

            for (int i = 0; i < rows; ++i)
            {
                var index = i * dimensions;

                if (results[index + confidenceIndex] <= 0.4f) continue;

                for (int j = labelStartIndex; j < dimensions; ++j)
                {
                    results[index + j] = results[index + j] * results[index + confidenceIndex];
                }

                for (int k = labelStartIndex; k < dimensions; ++k)
                {
                    if (results[index + k] <= 0.5f) continue;

                    var value_0 = results[index];
                    var value_1 = results[index + 1];
                    var value_2 = results[index + 2];
                    var value_3 = results[index + 3];

                    var bbox = new BBox((value_0 - value_2 / 2) / inputlWidth,
                                (value_1 - value_3 / 2) / inputlWidth,
                                (value_0 + value_2 / 2) / inputHeight,
                                (value_1 + value_3 / 2) / inputHeight);

                    var l_index = k - labelStartIndex;
                    detections.Add(new YoloPrediction()
                    {
                        Box = bbox,
                        Confidence = results[index + k],
                        LabelIndex = l_index,
                        LabelName = YoloPrediction.YoloLabes[l_index]
                    });


                }


            }


            return YoloPrediction.NMS(detections);

        }

    }



    //public class YOLOv5Result
    //{

    //    public static List<YOLOv5Result> ParseResults(float[] results)
    //    {
    //        //output  0 25200 85
    //        //box format  0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence
    //        int dimensions = 85;
    //        int rows = results.Length / dimensions;
    //        int confidenceIndex = 4;
    //        int labelStartIndex = 5;

    //        //float modelWidth = 640.0f;
    //        //float modelHeight = 640.0f;
    //        //float xGain = modelWidth / image.width;
    //        //float yGain = modelHeight / image.height;

    //        List<YOLOv5Result> detections = new List<YOLOv5Result>();

    //        for (int i = 0; i < rows; ++i)
    //        {
    //            var index = i * dimensions;

    //            if (results[index + confidenceIndex] <= 0.4f) continue;

    //            for (int j = labelStartIndex; j < dimensions; ++j)
    //            {
    //                results[index + j] = results[index + j] * results[index + confidenceIndex];
    //            }

    //            for (int k = labelStartIndex; k < dimensions; ++k)
    //            {
    //                if (results[index + k] <= 0.5f) continue;

    //                var value_0 = results[index];
    //                var value_1 = results[index + 1];
    //                var value_2 = results[index + 2];
    //                var value_3 = results[index + 3];

    //                var bbox = new BBox((value_0 - value_2 / 2) / 640,
    //                            (value_0 + value_2 / 2) / 640,
    //                            (value_1 - value_3 / 2) / 640,
    //                            (value_1 + value_3 / 2) / 640);

    //                detections.Add(new YOLOv5Result()
    //                {
    //                    bbox = bbox,
    //                    label_index = k - labelStartIndex,
    //                    prob = results[index + k]
    //                });


    //            }


    //        }



    //        return NMS(detections);

    //    }


    //    private static float ComputeIOU(YOLOv5Result DRa, YOLOv5Result DRb)
    //    {
    //        float ay1 = DRa.bbox.MinY;
    //        float ax1 = DRa.bbox.MinX;
    //        float ay2 = DRa.bbox.MaxY;
    //        float ax2 = DRa.bbox.MaxX;
    //        float by1 = DRb.bbox.MinY;
    //        float bx1 = DRb.bbox.MinX;
    //        float by2 = DRb.bbox.MaxY;
    //        float bx2 = DRb.bbox.MaxX;

    //        Debug.Assert(ay1 < ay2);
    //        Debug.Assert(ax1 < ax2);
    //        Debug.Assert(by1 < by2);
    //        Debug.Assert(bx1 < bx2);

    //        // determine the coordinates of the intersection rectangle
    //        float x_left = Math.Max(ax1, bx1);
    //        float y_top = Math.Max(ay1, by1);
    //        float x_right = Math.Min(ax2, bx2);
    //        float y_bottom = Math.Min(ay2, by2);

    //        if (x_right < x_left || y_bottom < y_top)
    //            return 0;
    //        float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    //        float bb1_area = (ax2 - ax1) * (ay2 - ay1);
    //        float bb2_area = (bx2 - bx1) * (by2 - by1);
    //        float iou = intersection_area / (bb1_area + bb2_area - intersection_area);

    //        Debug.Assert(iou >= 0 && iou <= 1);
    //        return iou;
    //    }


    //    private static List<YOLOv5Result> NMS(IReadOnlyList<YOLOv5Result> detections,
    //        float IOU_threshold = 0.45f,
    //        float score_threshold = 0.3f)
    //    {
    //        List<YOLOv5Result> final_detections = new List<YOLOv5Result>();

    //        for (int i = 0; i < detections.Count; i++)
    //        {
    //            int j = 0;
    //            for (j = 0; j < final_detections.Count; j++)
    //            {
    //                if (ComputeIOU(final_detections[j], detections[i]) > IOU_threshold)
    //                {
    //                    break;
    //                }
    //            }
    //            if (j == final_detections.Count)
    //            {
    //                final_detections.Add(detections[i]);
    //            }
    //        }
    //        return final_detections;

    //    }



    //    public int label_index;
    //    public BBox bbox;
    //    public double prob;


    //    public record BBox(float MinX, float MaxX, float MinY, float MaxY);

    //}



}
