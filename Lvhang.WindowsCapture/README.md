\*\*1.1.0 (2021/10/30) release note: \*\*

* fix a bug which the ToBitmap extension method not work on some hardware.
* add IsManual options
* add nextFrame action in OnFrameArrived to control if send next frame.

sample code:

```
        _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
        {
            //set minial ms between frames
            MinFrameInterval = 50,
            IsManual = true,
        });
        _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

        _captureSession.PickAndCapture();



        //CaptureSession_OnFrameArrived method
        private async void CaptureSession_OnFrameArrived(Windows.Graphics.Capture.Direct3D11CaptureFrame frame, Action nextFrame)
        {

            var bitmap = obj.ToBitmap();

            //do something

            nextFrame(); //enable to get next frame, if set IsManual to false, you don't have to call this method to get next frame.

        }
```

this project base on below projects:

```
        https://github.com/robmikh/WPFCaptureSample
        https://github.com/Microsoft/Windows.UI.Composition-Win32-Samples/tree/master/dotnet/WPF/ScreenCapture
```
