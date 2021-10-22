using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    interface IYOLODetector
    {
        ProcessDetail Inference(Bitmap img);

        Task<ProcessDetail> InferenceAsync(Bitmap bitmapOrg);

        void Stop();
    }


    public class ProcessDetail
    {
        public long ResizeBitmapCost { get; set; }

        public long BitmapToTensorCost { get; set; }

        public long InferenceCost { get; set; }

        public long DrawResultCost { get; set; }
    }
}
