#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <onnxruntime_cxx_api.h>

using namespace std;
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
	void Run();
	void SetImgPath(string path);
	cv::Mat GetResultImage();

private:
	bool use_cuda = false;
	int image_size = 640; 
	string model_path;
	string image_path = "";
	Ort::Env env;
	Ort::Session* session_;
	Ort::Value input_tensor_{ nullptr };
	cv::Mat resultImage;
};

