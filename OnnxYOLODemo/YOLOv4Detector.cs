using Lvhang.WindowsCapture;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    public class YOLOv4Detector: IYOLODetector
    {

        private InferenceSession _onnxSession;

        public YOLOv4Detector(string model_path)
        {
            // Session Options
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.EnableProfiling = true;
            //options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            //options.OptimizedModelFilePath = ".\\Models\\optimized\\opt_yolov4.onnx";
            //options.AppendExecutionProvider_OpenVINO(@"MYRIAD_FP16");
            options.AppendExecutionProvider_DML(0);
            //options.AppendExecutionProvider_CPU(0);

            // create inference session
            _onnxSession = new InferenceSession(model_path, options);

            var providers = OrtEnv.Instance().GetAvailableProviders();

        }


        public void Inference(Bitmap bitmapOrg, out ProcessTime ptime)
        {
            ptime = new ProcessTime();

            var sw = new Stopwatch();
            sw.Start();
            var new_image = bitmapOrg.Resize(416, 416);
            sw.Stop();
            ptime.ResizeBitmap = sw.ElapsedMilliseconds;


            sw.Reset(); sw.Start();
            var input_tensor = new_image.ToOnnxTensor_13hw();
            sw.Stop();
            ptime.BitmapToTensor = sw.ElapsedMilliseconds;


            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("input_1:0", input_tensor));


            sw.Reset(); sw.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            sw.Stop();
            ptime.Inference = sw.ElapsedMilliseconds;


            //return bitmapOrg;

        }

        public Task InferenceAsync(Bitmap srouce, ProcessTime ptime)
        {
            throw new NotImplementedException();
        }

        public void Stop()
        {
            _onnxSession.EndProfiling();
            _onnxSession.Dispose();
        }

    }
}
