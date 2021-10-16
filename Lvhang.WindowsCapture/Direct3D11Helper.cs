using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Windows.Graphics.DirectX.Direct3D11;
using WinRT;

namespace Lvhang.WindowsCapture
{
    public static class Direct3D11Helper
    {

        static Guid ID3D11Texture2D = new Guid("6f15aaf2-d208-4e89-9ab4-489535d34f9c");

        [ComImport]
        [Guid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")]
        [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        [ComVisible(true)]
        interface IDirect3DDxgiInterfaceAccess
        {
            IntPtr GetInterface([In] ref Guid iid);
        };

        [DllImport(
            "d3d11.dll",
            EntryPoint = "CreateDirect3D11DeviceFromDXGIDevice",
            SetLastError = true,
            CharSet = CharSet.Unicode,
            ExactSpelling = true,
            CallingConvention = CallingConvention.StdCall
            )]
        static extern UInt32 CreateDirect3D11DeviceFromDXGIDevice(IntPtr dxgiDevice, out IntPtr graphicsDevice);


        public static IDirect3DDevice CreateDevice()
        {
            return CreateDevice(false);
        }


        public static IDirect3DDevice CreateDevice(bool useWARP)
        {
            var d3dDevice = new SharpDX.Direct3D11.Device(
                useWARP ? SharpDX.Direct3D.DriverType.Software : SharpDX.Direct3D.DriverType.Hardware,
                SharpDX.Direct3D11.DeviceCreationFlags.BgraSupport);
            var device = CreateDirect3DDeviceFromSharpDXDevice(d3dDevice);
            return device;
        }


        public static IDirect3DDevice CreateDirect3DDeviceFromSharpDXDevice(SharpDX.Direct3D11.Device d3dDevice)
        {
            IDirect3DDevice device = null;

            // Acquire the DXGI interface for the Direct3D device.
            using (var dxgiDevice = d3dDevice.QueryInterface<SharpDX.DXGI.Device3>())
            {
                // Wrap the native device using a WinRT interop object.
                uint hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.NativePointer, out IntPtr pUnknown);

                if (hr == 0)
                {
                    device = MarshalInterface<IDirect3DDevice>.FromAbi(pUnknown);
                    Marshal.Release(pUnknown);
                }
            }

            return device;
        }


        public static SharpDX.Direct3D11.Texture2D CreateSharpDXTexture2D(IDirect3DSurface surface)
        {
            var access = surface.As<IDirect3DDxgiInterfaceAccess>();
            var d3dPointer = access.GetInterface(ID3D11Texture2D);
            var d3dSurface = new SharpDX.Direct3D11.Texture2D(d3dPointer);
            return d3dSurface;
        }

    }
}
