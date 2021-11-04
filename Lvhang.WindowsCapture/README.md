this package is for windows only, so please make sure your porject's TFM is later then net5.0-windows10.0.17763 to make sure install successfully.

\<TargetFramework>net5.0-windows10.0.17763\</TargetFramework>

**1.1.0 (2021/10/30) release note:**

* fix a bug which the ToBitmap extension method not work on some hardware.
* add IsManual options, by default IsManual is true and you need to call nextFrame to get next frame.
* add nextFrame action in OnFrameArrived to control if send next frame.
* wehn you call PickAndCapture, you can pass an action that will run after user choose a windows/desktop and before receive fist frame. 

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
