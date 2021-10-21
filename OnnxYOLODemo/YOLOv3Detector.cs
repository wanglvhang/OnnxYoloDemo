using Lvhang.WindowsCapture;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    public class YOLOv3Detector : IYOLODetector
    {

        private InferenceSession _onnxSession;


        public YOLOv3Detector(string model_path)
        {
            // Session Options
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            //options.EnableProfiling = true;
            //options.ProfileOutputPathPrefix = "yolov3_profile";

            //options.IntraOpNumThreads = 2;
            //options.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            //options.InterOpNumThreads = 6;
            //options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            //options.OptimizedModelFilePath = ".\\Models\\optimized\\opt_yolov3-10.onnx";
            //options.AppendExecutionProvider_CPU(0);


            options.AppendExecutionProvider_DML(0);


            // create inference session
            _onnxSession = new InferenceSession(model_path, options);
        }


        public void Inference(Bitmap bitmapOrg, out ProcessTime ptime)
        {
            ptime = new ProcessTime();

            var sw = new Stopwatch();
            sw.Start();
            var new_image = bitmapOrg.Resize(416, 416);
            sw.Stop();
            ptime.ResizeBitmap = sw.ElapsedMilliseconds;

            //this.PrintOutput($"resized image: width:{new_image.Width} height:{new_image.Height}");

            sw.Reset(); sw.Start();
            var input_tensor = new_image.ToOnnxTensor_13hw();
            sw.Stop();
            ptime.BitmapToTensor = sw.ElapsedMilliseconds;


            //Get the Image Shape
            var image_shape = new DenseTensor<float>(new[] { 1, 2 });
            image_shape[0, 0] = bitmapOrg.Height;
            image_shape[0, 1] = bitmapOrg.Width;

            // Setup inputs and outputs
            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("input_1", input_tensor));
            container.Add(NamedOnnxValue.CreateFromTensor("image_shape", image_shape));


            sw.Reset(); sw.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            sw.Stop();
            ptime.Inference = sw.ElapsedMilliseconds;


            sw.Reset(); sw.Start();
            //Post Processing Steps
            var resultsArray = results.ToArray();
            Tensor<float> boxes = resultsArray[0].AsTensor<float>();
            Tensor<float> scores = resultsArray[1].AsTensor<float>();
            int[] indices = resultsArray[2].AsTensor<int>().ToArray();

            var len = indices.Length / 3;
            var out_classes = new int[len];
            float[] out_scores = new float[len];

            var predictions = new List<Prediction>();
            var count = 0;
            for (int i = 0; i < indices.Length; i = i + 3)
            {
                out_classes[count] = indices[i + 1];
                out_scores[count] = scores[indices[i], indices[i + 1], indices[i + 2]];
                predictions.Add(new Prediction
                {
                    Box = new Box(boxes[indices[i], indices[i + 2], 1],
                                     boxes[indices[i], indices[i + 2], 0],
                                     boxes[indices[i], indices[i + 2], 3],
                                     boxes[indices[i], indices[i + 2], 2]),
                    Class = LabelMap.Labels[out_classes[count]],
                    Score = out_scores[count]
                });
                count++;
            }

            // Put boxes, labels and confidence on image and save for viewing
            System.Drawing.Font font = new Font("Arial", 24f, System.Drawing.FontStyle.Bold);

            //this.PrintOutput($"draw prediction:{p.Class}/{p.Score}  xmin:{p.Box.Xmin} xmax:{p.Box.Xmax} ymin:{p.Box.Ymin} ymax:{p.Box.Ymax} ");
            using (var g = Graphics.FromImage(bitmapOrg))
            {
                foreach (var p in predictions)
                {
                    g.DrawRectangle(new System.Drawing.Pen(System.Drawing.Brushes.Red, 6),
                        new System.Drawing.Rectangle((int)p.Box.Xmin,
                        (int)p.Box.Ymin,
                        (int)(p.Box.Xmax - p.Box.Xmin),
                        (int)(p.Box.Ymax - p.Box.Ymin)));

                    g.DrawString($"{p.Class}, {p.Score:0.00}", font, System.Drawing.Brushes.Red, new System.Drawing.PointF(p.Box.Xmin, p.Box.Ymin));
                }

            }

            sw.Stop();
            ptime.DrawResult = sw.ElapsedMilliseconds;


            //return bitmapOrg;
        }

        public void Stop()
        {
            _onnxSession.EndProfiling();
            _onnxSession.Dispose();
        }
    }
}
