using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Windows.Graphics.Capture;
using WinRT;

namespace Lvhang.WindowsCapture
{
    public static class CaptureHelper
    {
        //static readonly Guid GraphicsCaptureItemGuid = new Guid("79C3F95B-31F7-4EC2-A464-632EF5D30760");

        [ComImport]
        [Guid("3E68D4BD-7135-4D10-8018-9FB6D9F33FA1")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        [ComVisible(true)]
        interface IInitializeWithWindow
        {
            void Initialize(
                IntPtr hwnd);
        }


        public static void SetWindow(this GraphicsCapturePicker picker, IntPtr hwnd)
        {
            var interop = picker.As<IInitializeWithWindow>();
            interop.Initialize(hwnd);
        }

    }

}
