using Lvhang.WindowsCapture;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp.Extensions;
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
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

            var os = Environment.OSVersion;

            if (os.Platform == PlatformID.Win32NT && os.Version.Build < 18362)
            {
                MessageBox.Show("该程序需要Windows 10 Build 18362 及以上版本，请更新系统。");
                this.Close();
            }

        }






        private void btnStartCaptureYolov3_Click(object sender, RoutedEventArgs e)
        {
            this.StopCapture();

            this.PrintOutput("初始化窗口抓取组件...");
            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 0,
            });

            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture(() =>
            {
                this.PrintOutput("开始载入yolov3.onnx模型");
                var sw = new Stopwatch();
                sw.Start();
                this._yolovDetector = new YOLOv3Detector($".\\Models\\yolov3-10.onnx", true);
                sw.Stop();
                this.PrintOutput($"载入完成，花费{sw.ElapsedMilliseconds}ms");
            });

        }

        private void btnStartCaptureYolov4_Click(object sender, RoutedEventArgs e)
        {
            this.StopCapture();

            this.PrintOutput("初始化窗口抓取组件...");
            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 100,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture(()=> {
                this.PrintOutput("开始载入yolov4.onnx模型");
                var sw = new Stopwatch();
                sw.Start();
                this._yolovDetector = new YOLOv4Detector($".\\Models\\yolov4.onnx");
                sw.Stop();
                this.PrintOutput($"载入完成，花费{sw.ElapsedMilliseconds}ms");
            });
        }

        private void btnStartCaptureYolov5_Click(object sender, RoutedEventArgs e)
        {
            this.StopCapture();

            this._yolovDetector = new YOLOv5Detector($".\\Models\\yolov5n.onnx", true);

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 100,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();
        }

        private void btnStartCaptureYolov5_cpu_Click(object sender, RoutedEventArgs e)
        {
            this.StopCapture();

            this._yolovDetector = new YOLOv5Detector($".\\Models\\yolov5s.onnx", false);

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 100,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();
        }






        private void btnStopCapture_Click(object sender, RoutedEventArgs e)
        {
            this.StopCapture();
        }


        private long _frameCount = 0;
        private DateTime _frameStartTime;


        private async void CaptureSession_OnFrameArrived(Windows.Graphics.Capture.Direct3D11CaptureFrame frame, Action nextFrame)
        {
            //首帧
            if (_frameCount == 0)
            {
                _frameStartTime = DateTime.Now;
            }

            try
            {

                var sw = new Stopwatch();
                sw.Start();
                var source = frame.ToBitmap();
                sw.Stop();
                var ftb = sw.ElapsedMilliseconds;


                sw.Reset(); sw.Start();
                var pd = await this._yolovDetector.InferenceAsync(source);
                sw.Stop();
                var detect = sw.ElapsedMilliseconds;


                sw.Reset(); sw.Start();
                this.imgResult.Source = source.ToImageSource();
                sw.Stop();
                var showbitmap = sw.ElapsedMilliseconds;

                _frameCount++;

                var totalSeconds = (DateTime.Now - _frameStartTime).TotalSeconds;
                if (totalSeconds != 0)
                {
                    //PrintOutput($"FPS:{ _frameCount / totalSeconds }");
                    this.PrintOutput($"ftb:{ftb}ms|thread:<{pd.ThreadId}>|detect:{detect}ms(resize:{pd.ResizeCost}|totensor:{pd.ToTensorCost}|onnx:{pd.InferenceCost}|draw:{pd.DrawCost})|show:{showbitmap}ms[total:{ftb + detect + showbitmap}ms,avg FPS:{(_frameCount / totalSeconds).ToString("0.00") }]");
                }

                nextFrame();

            }
            catch (Exception ex)
            {
                _captureSession?.StopCapture();
                _captureSession = null;
                _yolovDetector = null;
                MessageBox.Show(ex.Message);
            }
            finally
            {

            }

        }


        public void PrintOutput(string line, bool newline = true)
        {

            //在输入内容前添加时间
            if (newline)
            {
                this.txbLog.AppendText("\r\n");
                this.txbLog.AppendText($"【{DateTime.Now.ToString("HH:mm:ss-fff")}】{line}");
            }
            else
            {
                this.txbLog.AppendText($"{line}");
            }
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

        public void StopCapture()
        {
            _frameCount = 0;
            _captureSession?.StopCapture();
        }


    }
}
