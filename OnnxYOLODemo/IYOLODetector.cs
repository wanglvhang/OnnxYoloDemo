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


    //public  class ProcessDetail2
    //{
    //    public long ResizeBitmapCost { get; set; }

    //    public long BitmapToTensorCost { get; set; }

    //    public long InferenceCost { get; set; }

    //    public long DrawResultCost { get; set; }
    //}


    public record ProcessDetail(int ThreadId,long ResizeCost, long ToTensorCost, long InferenceCost, long DrawCost);
}
