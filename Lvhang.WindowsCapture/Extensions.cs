using SharpDX;
using SharpDX.Direct3D11;
using SharpDX.DXGI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX.Direct3D11;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;
using WinRT;
using BitmapEncoder = Windows.Graphics.Imaging.BitmapEncoder;

namespace Lvhang.WindowsCapture
{
    public static class Extensions
    {

        public static Bitmap ToBitmap(this Direct3D11CaptureFrame frame)
        {
            var texture2d_bitmap = Direct3D11Helper.CreateSharpDXTexture2D(frame.Surface);

            var d3dDevice = texture2d_bitmap.Device;

            // Create texture copy
            var staging = new Texture2D(d3dDevice, new Texture2DDescription
            {
                Width = frame.ContentSize.Width,
                Height = frame.ContentSize.Height,
                MipLevels = 1,
                ArraySize = 1,
                Format = texture2d_bitmap.Description.Format,
                Usage = ResourceUsage.Staging,
                SampleDescription = new SampleDescription(1, 0),
                BindFlags = BindFlags.None,
                CpuAccessFlags = CpuAccessFlags.Read,
                OptionFlags = ResourceOptionFlags.None
            });

            try
            {
                // Copy data
                d3dDevice.ImmediateContext.CopyResource(texture2d_bitmap, staging);

                var dataBox = d3dDevice.ImmediateContext.MapSubresource(staging, 0, 0, MapMode.Read, SharpDX.Direct3D11.MapFlags.None,
                     out DataStream stream);

                //处理bitmap padding
                var bitmap_width = staging.Description.Width;
                var bitmap_stride = bitmap_width % 32 == 0 ? bitmap_width * 4 : (bitmap_width + (32 - (bitmap_width % 32))) * 4;

                var bitmap = new System.Drawing.Bitmap(staging.Description.Width, staging.Description.Height, bitmap_stride,
                     System.Drawing.Imaging.PixelFormat.Format32bppArgb, dataBox.DataPointer);

                return bitmap;
            }
            catch (Exception ex)
            {
                throw;
            }
            finally
            {
                staging.Dispose();
            }

        }


        public static Stream ToBitmapStream(this Direct3D11CaptureFrame frame)
        {
            var bitmap = frame.ToBitmap();
            Stream memoryStream = new MemoryStream();
            bitmap.Save(memoryStream, ImageFormat.Png);

            return memoryStream;
        }


        public static async Task<SoftwareBitmap> ToSoftwareBitmapAsync(this Direct3D11CaptureFrame frame)
        {
            var result = await SoftwareBitmap.CreateCopyFromSurfaceAsync(frame.Surface, BitmapAlphaMode.Premultiplied);

            return result;
        }


        public static async Task<Bitmap> ToBitmapAsync(this SoftwareBitmap sbitmap)
        {
            using (var stream = new Windows.Storage.Streams.InMemoryRandomAccessStream())
            {
                BitmapEncoder encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, stream);
                encoder.SetSoftwareBitmap(sbitmap);
                await encoder.FlushAsync();
                var bmp = new System.Drawing.Bitmap(stream.AsStream());
                return bmp;
            }

        }


        public static BitmapImage ToImageSource(this Bitmap bitmap)
        {
            MemoryStream ms = new MemoryStream();
            bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
            BitmapImage image = new BitmapImage();
            image.BeginInit();
            ms.Seek(0, SeekOrigin.Begin);
            image.StreamSource = ms;
            image.EndInit();

            return image;
        }


        public static Bitmap Resize(this Bitmap source, int new_width, int new_height)
        {

            float w_scale = (float)new_width / source.Width;
            float h_scale = (float)new_height / source.Height;

            float min_scale = Math.Min(w_scale, h_scale);

            var nw = (int)(source.Width * min_scale);
            var nh = (int)(source.Height * min_scale);


            var pad_dims_w = (new_width - nw) / 2;
            var pad_dims_h = (new_height - nh) / 2;


            var new_bitmap = new Bitmap(new_width, new_height, PixelFormat.Format24bppRgb);

            using (var g = Graphics.FromImage(new_bitmap))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighSpeed;
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Low;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;

                g.DrawImage(source, new Rectangle(pad_dims_w, pad_dims_h, nw, nh),
                    0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
            }

            return new_bitmap;
        }


        public static Bitmap ResizeWithoutPadding(this Bitmap source, int new_width, int new_height)
        {

            var new_bitmap = new Bitmap(new_width, new_height, PixelFormat.Format24bppRgb);

            using (var g = Graphics.FromImage(new_bitmap))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighSpeed;
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Low;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;

                g.DrawImage(source, new Rectangle(0, 0, new_width, new_height),
                    0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
            }

            return new_bitmap;

        }


    }
}
