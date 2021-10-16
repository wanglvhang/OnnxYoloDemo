using Lvhang.WindowsCapture;
using System;
using System.Diagnostics;
using System.Windows;

namespace OnnxYOLODemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        private WindowsCaptureSession _captureSession;
        private IYOLODetector _yolovDetector;

        public MainWindow()
        {
            InitializeComponent();
        }


        private void Window_Loaded(object sender, RoutedEventArgs e)
        {

        }


        private async void btnStartCaptureYolov3_Click(object sender, RoutedEventArgs e)
        {
            this._yolovDetector = new YOLOv3Detector($".\\Models\\yolov3-10.onnx");

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 50,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();

        }

        private void btnStartCaptureYolov4_Click(object sender, RoutedEventArgs e)
        {
            this._yolovDetector = new YOLOv4Detector($".\\Models\\yolov4.onnx");

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 50,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();
        }



        private void btnStopCapture_Click(object sender, RoutedEventArgs e)
        {
            _captureSession.StopCapture();
        }


        private async void CaptureSession_OnFrameArrived(Windows.Graphics.Capture.Direct3D11CaptureFrame obj)
        {

            var sw = new Stopwatch();
            sw.Start();
            var bitmap = obj.ToBitmap();
            sw.Stop();
            this.PrintOutput($"frame to bitmap cost {sw.ElapsedMilliseconds}ms");



            sw.Reset();sw.Start();
            var result_bitmap = this._yolovDetector.Inference(bitmap,out ProcessTime ptime);
            sw.Stop();
            this.PrintOutput($"Inference cost {sw.ElapsedMilliseconds}ms, resize:{ptime.ResizeBitmap}|totensor:{ptime.BitmapToTensor}|onnx:{ptime.Inference}|draw:{ptime.DrawResult}");
            sw.Reset();


            this.imgResult.Source = bitmap.ToImageSource();



        }


        public void PrintOutput(string line, bool newline = true)
        {
            //在输入内容前添加时间
            if (newline) this.txbLog.AppendText("\r\n");
            this.txbLog.AppendText($"【{DateTime.Now.ToString("HH:mm:ss-fff")}】{line}");
            this.txbLog.ScrollToEnd();

            //控制行数， 每次达到1000行时，删除前300行
            if (this.txbLog.LineCount >= 1000)
            {
                this.txbLog.AppendText("\r\n");
                this.txbLog.AppendText($"【{DateTime.Now.ToString("HH:mm:ss-fff")}】清理前300行日志.");

                int end_index = 0;
                int line_index = 0;
                while (line_index < 300)
                {
                    end_index += this.txbLog.GetLineLength(line_index);
                    line_index++;
                }
                this.txbLog.Text = this.txbLog.Text.Remove(0, end_index);
            }

        }

    }
}
