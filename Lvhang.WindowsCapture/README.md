sample code:

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                //set minial ms between frames
                MinFrameInterval = 50,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();
            
            
            
            //CaptureSession_OnFrameArrived method
            private async void CaptureSession_OnFrameArrived(Windows.Graphics.Capture.Direct3D11CaptureFrame obj)
            {

                var bitmap = obj.ToBitmap();

            }
            
            


this project base on below projects:
            https://github.com/robmikh/WPFCaptureSample
            https://github.com/Microsoft/Windows.UI.Composition-Win32-Samples/tree/master/dotnet/WPF/ScreenCapture
