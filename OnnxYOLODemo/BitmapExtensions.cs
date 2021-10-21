using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace OnnxYOLODemo
{
    public static class BitmapExtensions
    {

        //for yolov3
        public static Tensor<float> ToOnnxTensor_13hw(this Bitmap bitmap)
        {

            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, bitmap.Height, bitmap.Width });

            var db = new DirectReadBitmap(bitmap);

            //读取bitmap 像素  bytes [B,G,R,A]
            Parallel.For(0, bitmap.Height, (idx, state) =>
            {

                WriteBitmapToTensor(db, tensor, idx);

            });

            return tensor;

        }


        public static void WriteBitmapToTensor(DirectReadBitmap directbitmap, Tensor<float> tensor, int row)
        {

            for (int x = 0; x < directbitmap.Width; x++)
            {
                var pixel = directbitmap.GetPixel(x, row);

                tensor[0, 0, row, x] = pixel.B / 255f;
                tensor[0, 1, row, x] = pixel.G / 255f;
                tensor[0, 2, row, x] = pixel.R / 255f;

            }

        }


        //for yolov4
        public static Tensor<float> ToOnnxTensor_1hw3(this Bitmap bitmap)
        {
            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, bitmap.Height, bitmap.Width, 3 });

            var db = new DirectReadBitmap(bitmap);

            //读取bitmap 像素  bytes [B,G,R,A]
            for (int x = 0; x < bitmap.Width; x++)
            {
                for (int y = 0; y < bitmap.Width; y++)
                {
                    var pixel = db.GetPixel(x, y);

                    tensor[0, y, x, 0] = pixel.B / 255f;
                    tensor[0, y, x, 1] = pixel.G / 255f;
                    tensor[0, y, x, 2] = pixel.R / 255f;
                }
            }

            return tensor;
        }


        //提升了15ms左右
        public static Tensor<float> ToOnnxTensorUnsafe_13hw(this Bitmap image)
        {
            // Create the Tensor with the appropiate dimensions  for the NN
            Tensor<float> data = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });

            BitmapData bmd = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            int PixelSize = 4;

            unsafe
            {
                for (int y = 0; y < bmd.Height; y++)
                {
                    // row is a pointer to a full row of data with each of its colors
                    byte* row = (byte*)bmd.Scan0 + (y * bmd.Stride);
                    for (int x = 0; x < bmd.Width; x++)
                    {
                        // note the order of colors is BGR
                        data[0, 0, y, x] = row[x * PixelSize + 2] / (float)255.0;
                        data[0, 1, y, x] = row[x * PixelSize + 1] / (float)255.0;
                        data[0, 2, y, x] = row[x * PixelSize + 0] / (float)255.0;
                    }
                }

                image.UnlockBits(bmd);
            }
            return data;
        }


    }

}
