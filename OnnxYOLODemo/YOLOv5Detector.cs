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
                options.OptimizedModelFilePath = ".\\Models\\optimized\\opt_yolov5m.onnx";
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



}
