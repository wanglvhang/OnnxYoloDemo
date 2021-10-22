using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OnnxYOLODemo
{
    //参考自: https://stackoverflow.com/questions/24701703/c-sharp-faster-alternatives-to-setpixel-and-getpixel-for-bitmaps-for-windows-f
    public class DirectReadBitmap : IDisposable
    {
        public Bitmap Bitmap { get; private set; }
        public Color[] Pixels { get; private set; }
        public bool Disposed { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }


        public DirectReadBitmap(Bitmap source)
        {
            this.Bitmap = source;
            this.Width = source.Width;
            this.Height = source.Height;

            Pixels = new Color[source.Width * source.Height];

            var bitmap_data = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            var bitmap_bytes = new byte[Math.Abs(bitmap_data.Stride) * source.Height];

            //copy bytes to pixels
            Marshal.Copy(bitmap_data.Scan0, bitmap_bytes, 0, bitmap_bytes.Length);

            Parallel.For(0, Pixels.Length, (p_idx, state) => {

                var color = Color.FromArgb(bitmap_bytes[p_idx * 3 + 2], bitmap_bytes[p_idx * 3 + 1], bitmap_bytes[p_idx*3]);
                Pixels[p_idx] = color;

            });


            source.UnlockBits(bitmap_data);

            //var stream = new MemoryStream(bitmap_bytes);
            //var bmp = new Bitmap(stream);
            //bmp.Save($".\\result_images\\{new Random().Next()}.png");

        }


        public Color GetPixel(int x, int y)
        {
            int index = x + (y * Width);

            return Pixels[index];
        }

        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;
            Bitmap.Dispose();
        }

    }
}
