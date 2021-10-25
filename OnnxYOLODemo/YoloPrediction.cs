using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace OnnxYOLODemo
{
    public class YoloPrediction
    {
        public BBox Box { get; set; }

        public int LabelIndex { get; set; }

        public string LabelName { get; set; }

        public float Confidence { get; set; }


        public static List<YoloPrediction> NMS(List<YoloPrediction> predictions, float IOU_threshold = 0.45f, float score_threshold = 0.3f)
        {
            List<YoloPrediction> final_predications = new List<YoloPrediction>();

            for (int i = 0; i < predictions.Count; i++)
            {
                int j = 0;
                for (j = 0; j < final_predications.Count; j++)
                {
                    if (ComputeIOU(final_predications[j], predictions[i]) > IOU_threshold)
                    {
                        break;
                    }
                }
                if (j == final_predications.Count)
                {
                    final_predications.Add(predictions[i]);
                }
            }
            return final_predications;
        }


        private static float ComputeIOU(YoloPrediction DRa, YoloPrediction DRb)
        {
            float ay1 = DRa.Box.MinY;
            float ax1 = DRa.Box.MinX;
            float ay2 = DRa.Box.MaxY;
            float ax2 = DRa.Box.MaxX;
            float by1 = DRb.Box.MinY;
            float bx1 = DRb.Box.MinX;
            float by2 = DRb.Box.MaxY;
            float bx2 = DRb.Box.MaxX;

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


        public static readonly string[] YoloLabes = new string[] {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
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
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
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
            "toothbrush" };


    }


    public record BBox(float MinX, float MinY, float MaxX,  float MaxY);

}
