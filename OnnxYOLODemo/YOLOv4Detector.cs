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
            var ptime = new ProcessDetail();

            var sw = new Stopwatch();
            sw.Start();
            var new_image = bitmapOrg.ResizeWithoutPadding(416, 416);
            sw.Stop();
            ptime.ResizeBitmapCost = sw.ElapsedMilliseconds;


            sw.Reset(); sw.Start();
            var input_tensor = new_image.FastToOnnxTensor_13hw();
            sw.Stop();
            ptime.BitmapToTensorCost = sw.ElapsedMilliseconds;


            var container = new List<NamedOnnxValue>();
            container.Add(NamedOnnxValue.CreateFromTensor("input_1:0", input_tensor));


            sw.Reset(); sw.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _onnxSession.Run(container);
            sw.Stop();
            ptime.InferenceCost = sw.ElapsedMilliseconds;


            sw.Reset(); sw.Start();
            var resultsArray = results.ToArray();
            Tensor<float> tensors = resultsArray[0].AsTensor<float>();

            var detectioinResults = DetectionResult.ParseResults(tensors.ToArray());

            System.Drawing.Font font = new Font("Arial", 24f, System.Drawing.FontStyle.Bold);

            using (var g = Graphics.FromImage(bitmapOrg))
            {
                foreach (DetectionResult p in detectioinResults)
                {
                    int top = (int)(p.bbox[0] * bitmapOrg.Height);
                    int left = (int)(p.bbox[1] * bitmapOrg.Width);
                    int bottom = (int)(p.bbox[2] * bitmapOrg.Height);
                    int right = (int)(p.bbox[3] * bitmapOrg.Width);


                    g.DrawRectangle(new System.Drawing.Pen(System.Drawing.Brushes.Blue, 4),
                        new System.Drawing.Rectangle(left, top, right - left, bottom - top));

                    g.DrawString($"{p.label}, {p.prob:0.00}", font, System.Drawing.Brushes.Blue, new System.Drawing.PointF(left, top));
                }

            }
            sw.Stop();
            ptime.DrawResultCost = sw.ElapsedMilliseconds;

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
            _onnxSession?.EndProfiling();
            _onnxSession?.Dispose();
        }


    }



    public class DetectionResult
    {
        private static readonly string[] _labels =
         {
                "person",
                "bicycle",
                "car",
                "motorbike",
                "aeroplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "sofa",
                "pottedplant",
                "bed",
                "diningtable",
                "toilet",
                "tvmonitor",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush"
        };
        public static  List<DetectionResult> ParseResults(float[] results)
        {
            int c_values = 84;
            int c_boxes = results.Length / c_values;
            float confidence_threshold = 0.5f;
            List<DetectionResult> detections = new List<DetectionResult>();
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
                    List<float> bbox = new List<float>();
                    bbox.Add(results[i_box * c_values + 0]);
                    bbox.Add(results[i_box * c_values + 1]);
                    bbox.Add(results[i_box * c_values + 2]);
                    bbox.Add(results[i_box * c_values + 3]);

                    detections.Add(new DetectionResult()
                    {
                        label = _labels[label_index],
                        bbox = bbox,
                        prob = max_prob
                    });
                }
            }

            // Non-maximum Suppression(NMS), a technique which filters the proposals 
            // based on Intersection over Union(IOU)
            return NMS(detections);
        }

        private static float ComputeIOU(DetectionResult DRa, DetectionResult DRb)
        {
            float ay1 = DRa.bbox[0];
            float ax1 = DRa.bbox[1];
            float ay2 = DRa.bbox[2];
            float ax2 = DRa.bbox[3];
            float by1 = DRb.bbox[0];
            float bx1 = DRb.bbox[1];
            float by2 = DRb.bbox[2];
            float bx2 = DRb.bbox[3];

            Debug.Assert(ay1 < ay2);
            Debug.Assert(ax1 < ax2);
            Debug.Assert(by1 < by2);
            Debug.Assert(bx1 < bx2);

            // determine the coordinates of the intersection rectangle
            float x_left = Math.Max(ax1, bx1);
            float y_top = Math.Max(ay1, by1);
            float x_right = Math.Min(ax2, bx2);
            float y_bottom = Math.Min(ay2, by2);

            if (x_right < x_left || y_bottom < y_top)
                return 0;
            float intersection_area = (x_right - x_left) * (y_bottom - y_top);
            float bb1_area = (ax2 - ax1) * (ay2 - ay1);
            float bb2_area = (bx2 - bx1) * (by2 - by1);
            float iou = intersection_area / (bb1_area + bb2_area - intersection_area);

            Debug.Assert(iou >= 0 && iou <= 1);
            return iou;
        }


        private static List<DetectionResult> NMS(IReadOnlyList<DetectionResult> detections,
            float IOU_threshold = 0.45f,
            float score_threshold = 0.3f)
        {
            List<DetectionResult> final_detections = new List<DetectionResult>();
            for (int i = 0; i < detections.Count; i++)
            {
                int j = 0;
                for (j = 0; j < final_detections.Count; j++)
                {
                    if (ComputeIOU(final_detections[j], detections[i]) > IOU_threshold)
                    {
                        break;
                    }
                }
                if (j == final_detections.Count)
                {
                    final_detections.Add(detections[i]);
                }
            }
            return final_detections;
        }


        public string label;
        public List<float> bbox;
        public double prob;
    }
}
