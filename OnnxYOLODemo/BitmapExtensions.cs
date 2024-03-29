﻿using Microsoft.ML.OnnxRuntime.Tensors;
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
using Windows.Graphics.Capture;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;
using System.Runtime.InteropServices.WindowsRuntime;

namespace OnnxYOLODemo
{
    public static class BitmapExtensions
    {

        public static Tensor<float> ToOnnxTensor_13hw(this Bitmap bitmap)
        {

            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, bitmap.Height, bitmap.Width });

            var db = new DirectReadBitmap(bitmap);

            //读取bitmap 像素  bytes [B,G,R,A]
            Parallel.For(0, bitmap.Height, (idx, state) =>
            {

                //WriteBitmapToTensor(db, tensor, idx);

                for (int x = 0; x < db.Width; x++)
                {
                    var pixel = db.GetPixel(x, idx);

                    tensor[0, 0, idx, x] = pixel.B / 255f;
                    tensor[0, 1, idx, x] = pixel.G / 255f;
                    tensor[0, 2, idx, x] = pixel.R / 255f;
                }

            });

            return tensor;

        }

        public static Tensor<float> ToOnnxTensor_13wh(this Bitmap bitmap)
        {

            Tensor<float> tensor = new DenseTensor<float>(new[] { 1, 3, bitmap.Width, bitmap.Height });

            var db = new DirectReadBitmap(bitmap);

            //读取bitmap 像素  bytes [B,G,R,A]
            Parallel.For(0, bitmap.Height, (idx, state) =>
            {

                //WriteBitmapToTensor(db, tensor, idx);

                for (int x = 0; x < db.Width; x++)
                {
                    var pixel = db.GetPixel(x, idx);

                    tensor[0, 0, x, idx] = pixel.B / 255f;
                    tensor[0, 1, x, idx] = pixel.G / 255f;
                    tensor[0, 2, x, idx] = pixel.R / 255f;
                }

            });

            return tensor;

        }


        public static async Task<SoftwareBitmap> ToSoftwareBitmap(this Direct3D11CaptureFrame frame)
        {
            var sb = await SoftwareBitmap.CreateCopyFromSurfaceAsync(frame.Surface);

            return sb;

        }


        public static async Task<byte[]> ToBytes(this SoftwareBitmap sbitmap)
        {

            byte[] array = null;

            // First: Use an encoder to copy from SoftwareBitmap to an in-mem stream (FlushAsync)
            // Next:  Use ReadAsync on the in-mem stream to get byte[] array

            using (var ms = new InMemoryRandomAccessStream())
            {
                BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, ms);
                encoder.SetSoftwareBitmap(sbitmap);

                try
                {
                    await encoder.FlushAsync();
                }
                catch (Exception ex) { return new byte[0]; }

                array = new byte[ms.Size];
                await ms.ReadAsync(array.AsBuffer(), (uint)ms.Size, InputStreamOptions.None);
            }
            return array;

        }


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


        //RRRR GGGG BBBB
        public static Tensor<float> FastToOnnxTensor_13hw(this Bitmap source)
        {
            var floatArray = new float[source.Width * source.Height * 3];

            var bitmap_data = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            var bitmap_bytes = new byte[Math.Abs(bitmap_data.Stride) * source.Height];

            Marshal.Copy(bitmap_data.Scan0, bitmap_bytes, 0, bitmap_bytes.Length);

            int total_pixels_count = source.Width * source.Height;


            Parallel.For(0, total_pixels_count, (p_idx, state) =>
            {

                var g_idx = p_idx + total_pixels_count;
                var b_idx = p_idx + total_pixels_count * 2;

                floatArray[p_idx] = bitmap_bytes[p_idx * 3 + 2] / 255f;//R
                floatArray[g_idx] = bitmap_bytes[p_idx * 3 + 1] / 255f;//G
                floatArray[b_idx] = bitmap_bytes[p_idx * 3] / 255f;//B

            });

            source.UnlockBits(bitmap_data);

            return new DenseTensor<float>(new Memory<float>(floatArray), new int[] { 1, 3, source.Height, source.Width });

        }


        public static void DrawYoloPrediction(this Bitmap source, List<YoloPrediction> predictions,  System.Drawing.Brush boxColor, System.Drawing.Brush labelColor)
        {
            if (predictions is null || predictions.Count == 0) return;

            System.Drawing.Font font = new Font("Arial", 22f, System.Drawing.FontStyle.Regular);
            using (var g = Graphics.FromImage(source))
            {
                foreach(var p in predictions)
                {

                    int top = (int)(p.Box.MinY * source.Height);
                    int left = (int)(p.Box.MinX * source.Width);
                    int bottom = (int)(p.Box.MaxY * source.Height);
                    int right = (int)(p.Box.MaxX * source.Width);

                    g.DrawRectangle(new System.Drawing.Pen(boxColor, 3),
                        new System.Drawing.Rectangle(left, top, right - left, bottom - top));

                    g.DrawString($"{p.LabelName}, {p.Confidence:0.00}", font, labelColor, new System.Drawing.PointF(left, top));

                }
            }

        }



    }

}
