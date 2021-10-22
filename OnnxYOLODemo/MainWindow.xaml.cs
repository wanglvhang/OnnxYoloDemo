using Lvhang.WindowsCapture;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp.Extensions;
using System;
using System.Diagnostics;
using System.Drawing;
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
        //private Thread _workThread;
        //private Task<Bitmap> _workTask;
        //private SerialQueue _taskQueue;
        //private OpenCvSharp.Window cvWindow;

        public MainWindow()
        {
            InitializeComponent();
        }


        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            //this.cvWindow = new OpenCvSharp.Window();

            //var array = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
            //var ms = new Memory<float>(array);

            //var tensor = new DenseTensor<float>(ms, new int[] { 1, 2, 3, 3 });

            var os = Environment.OSVersion;

            if(os.Version.Build < 18362)
            {
                MessageBox.Show("该程序需要Windows 10 Build 18362 以上版本，请更新系统。");
                this.Close();
            }

        }


        private void btnStartCaptureYolov3_Click(object sender, RoutedEventArgs e)
        {
            this._captureSession?.StopCapture();

            this._yolovDetector = new YOLOv3Detector($".\\Models\\yolov3-10.onnx",true);

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 110,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();

        }

        private void btnStartCaptureYolov4_Click(object sender, RoutedEventArgs e)
        {
            this._captureSession?.StopCapture();

            this._yolovDetector = new YOLOv4Detector($".\\Models\\yolov4.onnx");

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 100,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();
        }


        private void btnStartCaptureYolov5_Click(object sender, RoutedEventArgs e)
        {
            this._captureSession?.StopCapture();

            this._yolovDetector = new YOLOv5Detector($".\\Models\\yolov5s.onnx", true);

            _captureSession = new WindowsCaptureSession(this, new WindowsCaptureSessionOptions()
            {
                MinFrameInterval = 100,
            });
            _captureSession.OnFrameArrived += CaptureSession_OnFrameArrived;

            _captureSession.PickAndCapture();
        }


        private void btnStartCaptureYolov5_cpu_Click(object sender, RoutedEventArgs e)
        {
            this._captureSession?.StopCapture();

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
            _captureSession?.StopCapture();
        }


        private async void CaptureSession_OnFrameArrived(Windows.Graphics.Capture.Direct3D11CaptureFrame frame)
        {
            try
            {

                var sw = new Stopwatch();
                sw.Start();
                var source = frame.ToBitmap();
                sw.Stop();
                //this.PrintOutput($"frameToBitmap:{sw.ElapsedMilliseconds}ms", true);
                var ftb = sw.ElapsedMilliseconds;


                sw.Reset(); sw.Start();
                var ptime = await this._yolovDetector.InferenceAsync(source);
                sw.Stop();
                var detect = sw.ElapsedMilliseconds;
                //this.PrintOutput($"Inference:{sw.ElapsedMilliseconds}ms, resize:{ptime.ResizeBitmapCost}|totensor:{ptime.BitmapToTensorCost}|onnx:{ptime.InferenceCost}|draw:{ptime.DrawResultCost}", false); ;


                sw.Reset(); sw.Start();
                this.imgResult.Source = source.ToImageSource();
                sw.Stop();
                var showbitmap = sw.ElapsedMilliseconds;
                //this.PrintOutput($"show:{sw.ElapsedMilliseconds}ms", false);

                this.PrintOutput($"ftb:{ftb}ms|detect:{detect}ms(resize:{ptime.ResizeBitmapCost}|totensor:{ptime.BitmapToTensorCost}|onnx:{ptime.InferenceCost}|draw:{ptime.DrawResultCost})|show:{showbitmap}ms [total:{ftb+detect+showbitmap}ms]");

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

    }
}
