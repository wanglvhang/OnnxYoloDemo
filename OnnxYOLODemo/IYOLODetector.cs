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


    public record ProcessDetail(int ThreadId,long ResizeCost, long ToTensorCost, long InferenceCost, long DrawCost);


}
