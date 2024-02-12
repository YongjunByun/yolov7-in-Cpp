## 소개
실시간 Object Detection 모델인 YOLOv7을 ONNX C++ 사용 튜토리얼을 참고하여 윈도우 어플리케이션에서 사용 가능하게 만들어보고 나중에 참고하려고 만들었습니다.

## reference :
- [onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)
- [Easiest way to Train yolov7 on the custom dataset – 2024](https://machinelearningprojects.net/train-yolov7-on-the-custom-dataset/)
- [Using YOLO in C++](https://medium.com/@shahriar.rezghi.sh/using-yolo-in-c-55d55419a947)

***************
### Step1 - 환경구성
- YOLOv7 공식 레포지토리를 clone하고 requirements.txt를 설치합니다.

### Step2 - 데이터셋 준비
- data 폴더에 **"train", "val" 폴더를 생성**후 이미지와 라벨 파일을 넣어줍니다.   
<img src="https://github.com/YongjunByun/yolov7-in-Cpp/assets/82483513/5c402900-cfa6-4603-a098-3adb686bdebd" height="250">

  
### Step3 - Config 파일 수정
- **"coco.yaml"** 파일을 열고 아래와 같이 수정합니다.
> 1. **'train: data/train'** 
> 2. **'val: data/val'**
> 3. **'nc:1'** (no of classes), 데이터셋에 알맞게 수정합니다.
> 4. **names:[‘Package’]** 데이터셋에 알맞게 수정합니다.

- **yolov7/cfg/training** 경로에 있는 **"yolov7.yaml"** 파일을 열고 **nc** 를 Step3에서 설정한 nc와 동일하게 수정합니다.

### Step4 - pretrained wights 파일 다운로드
- [https://github.com/WongKinYiu/yolov7#performance](https://github.com/WongKinYiu/yolov7#performance)에서 YOLOv7 다운로드합니다.
<img src="https://github.com/YongjunByun/yolov7-in-Cpp/assets/82483513/6eff5610-7764-4877-83ce-afa723c7c26a" height="250">  

- **yolov7.pt** 파일을 yolov7 폴더로 이동합니다.

### Step5 - Train
```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -r drive/MyDrive/yolov7/requirements.txt
!pip install -r drive/MyDrive/yolov7/requirements_gpu.txt

!python train.py --workers 1 --device 0 --batch-size 16 --epochs 100 --img 640 640 --hyp data/hyp.scratch.  custom.yaml --name yolov7-custom --weights yolov7.pt
```
- colab에서 실행하였습니다. 
- 모델 테스트는 생략합니다

### Step6 - ONNX로 Export
- **export.py**로 생성된 **.pt*파일을 **.onnx**로 변환합니다.
```python
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --include-nms \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

### Step7 - C++ 프로그래밍
- **OpenCV**와 **ONNX Runtime**을 설치합니다.
<img src="https://github.com/YongjunByun/yolov7-in-Cpp/assets/82483513/aebbcf4c-4df3-421e-9b2c-bb47520e0ef7" height="250">

- 모델을 GPU로 돌리려는 경우 **CUDA Libraries**도 필요합니다.
- LoadDnnModel 클래스(계속 수정) :
  
```c++
//LoadDnnModel.h
using Array = std::vector<float>;
using Shape = std::vector<int64_t>;
class LoadDnnModel
{
public:
	LoadDnnModel();
	~LoadDnnModel();
	void LoadOnnx();
	pair<Array, Shape> process_image(Ort::Session* session, Array& array, Shape shape);
	tuple<Array, Shape, cv::Mat> Read_Image(const string& path, int size);
	void drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid);
	void display_image(cv::Mat image, const Array& output, const Shape& shape);

private:
	bool use_cuda = false;
	int image_size = 640; 
	string model_path = "weights\\best.onnx";
	string image_path = "";
	Ort::Env env;
	Ort::Session* session_;
	Ort::Value input_tensor_{ nullptr };
};
```
- classes의 이름도 설정해야 합니다.
```c++
//LoadDnnModel.cpp
const char* class_names[] = { "packages" };
```

### Step8 - Inference Output
