using System;
using System.Windows;
using System.Windows.Interop;
using Windows.Graphics;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX;
using Windows.Graphics.DirectX.Direct3D11;

namespace Lvhang.WindowsCapture
{
    public class WindowsCaptureSession : IDisposable
    {
        public WindowsCaptureSession(Window mainWindow, WindowsCaptureSessionOptions options)
        {
            this._options = options;
            var interopWindow = new WindowInteropHelper(mainWindow);
            _window_hwnd = interopWindow.Handle;
            _device = Direct3D11Helper.CreateDevice();
#if DEBUG
            // force graphicscapture.dll to load
            this._picker = new GraphicsCapturePicker();
#endif
            _last_frame_send_time = DateTime.Now;
        }

        private GraphicsCapturePicker _picker;
        private Direct3D11CaptureFramePool _framePool;
        private IntPtr _window_hwnd;
        private IDirect3DDevice _device;
        private GraphicsCaptureSession _session;
        private WindowsCaptureSessionOptions _options;

        private SizeInt32 _lastSize;
        private DateTime _last_frame_send_time;

        public void Dispose()
        {
            _device.Dispose();
        }

        public event Action<Direct3D11CaptureFrame> OnFrameArrived;

        public async void PickAndCapture()
        {
            try
            {

                _picker = new GraphicsCapturePicker();
                _picker.SetWindow(_window_hwnd);

                var item = await _picker.PickSingleItemAsync();

                if (item != null)
                {
                    //_d3dDevice = Direct3D11Helper.CreateSharpDXDevice(_device);
                    //_capture_app_sample.StartCaptureFromItem(item, _processor);

                    _framePool = Direct3D11CaptureFramePool.Create(
                                            _device,
                                            DirectXPixelFormat.B8G8R8A8UIntNormalized,
                                            2,
                                            item.Size);

                    _session = _framePool.CreateCaptureSession(item);

                    _framePool.FrameArrived += _framePool_FrameArrived;

                    _session.StartCapture();
                }
            }
            catch (Exception ex)
            {
                throw;
            }
        }

        private void _framePool_FrameArrived(Direct3D11CaptureFramePool sender, object args)
        {
            var newSize = false;
            using (var frame = sender.TryGetNextFrame())
            {
                if (frame.ContentSize.Width != _lastSize.Width || frame.ContentSize.Height != _lastSize.Height)
                {
                    newSize = true;
                    _lastSize = frame.ContentSize;
                }

                if ((DateTime.Now - _last_frame_send_time).TotalMilliseconds >= _options.MinFrameInterval)
                {
                    var bitmap_surface = Direct3D11Helper.CreateSharpDXTexture2D(frame.Surface);

                    OnFrameArrived(frame);
                    _last_frame_send_time = DateTime.Now;
                }

            }

            if (newSize)
            {
                _framePool.Recreate(
                    _device,
                    DirectXPixelFormat.B8G8R8A8UIntNormalized,
                    2,
                    _lastSize);
            }

        }

        public void StopCapture()
        {
            _framePool?.Dispose();
            _session?.Dispose();
            _device?.Dispose();
        }
    }

}
