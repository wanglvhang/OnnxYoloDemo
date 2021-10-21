using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    interface IYOLODetector
    {
        void Inference(Bitmap srouce,out ProcessTime ptime);

        void Stop();
    }


    public class ProcessTime
    {
        public long ResizeBitmap { get; set; }

        public long BitmapToTensor { get; set; }

        public long Inference { get; set; }

        public long DrawResult { get; set; }
    }
}
