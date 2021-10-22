using Lvhang.WindowsCapture;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    public class YOLOv5Detector : IYOLODetector
    {

        private InferenceSession _onnxSession;
        private Mutex _sessionMutex = new Mutex();

        public YOLOv5Detector(string model_path,bool useDirectML)
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

            var ptime = new ProcessDetail();

            var sw = new Stopwatch();
            sw.Start();
            var resized_image = img.Resize(640, 640);
            sw.Stop();
            ptime.ResizeBitmapCost = sw.ElapsedMilliseconds;


            sw.Reset(); sw.Start();
            var input_tensor = resized_image.FastToOnnxTensor_13hw();
            sw.Stop();
            ptime.BitmapToTensorCost = sw.ElapsedMilliseconds;


            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("images", input_tensor));


            sw.Reset(); sw.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            sw.Stop();
            ptime.InferenceCost = sw.ElapsedMilliseconds;

            return ptime;

        }

        public Task<ProcessDetail> InferenceAsync(Bitmap bitmapOrg)
        {
            return Task.Run(() =>
            {
                ProcessDetail ptime = null;
                if (this._sessionMutex.WaitOne())
                {
                    ptime = Inference(bitmapOrg);
                    this._sessionMutex.ReleaseMutex();

                }
                return ptime;

            });
        }

        public void Stop()
        {
            //_onnxSession?.EndProfiling();
            _onnxSession?.Dispose();
        }
    }
}
