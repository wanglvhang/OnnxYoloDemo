using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace OnnxYOLODemo
{
    interface IYOLODetector
    {
        Bitmap Inference(Bitmap srouce,out ProcessTime ptime);
    }


    public class ProcessTime
    {
        public long ResizeBitmap { get; set; }

        public long BitmapToTensor { get; set; }

        public long Inference { get; set; }

        public long DrawResult { get; set; }
    }
}
