#include "LoadDnnModel.h"

const char* class_names[] = { "package" };

LoadDnnModel::LoadDnnModel()
{
	model_path = "weights\\best.onnx";
}

LoadDnnModel::~LoadDnnModel()
{

}

tuple<Array, Shape, cv::Mat> LoadDnnModel::Read_Image(const string& path, int size)
{
    auto image = cv::imread(path);
    assert(!image.empty() && image.channels() == 3);
    cv::resize(image, image, { size, size });
    Shape shape = { 1, image.channels(), image.rows, image.cols };
    cv::Mat nchw = cv::dnn::blobFromImage(image, 1.0, {}, {}, true) / 255.f;
    Array array(nchw.ptr<float>(), nchw.ptr<float>() + nchw.total());
    return { array, shape, image };
}
void LoadDnnModel::SetImgPath(string path) {
	image_path = path;
}
void LoadDnnModel::Run() {
	tuple<Array, Shape, cv::Mat> temp = Read_Image(image_path, 640);
	pair<Array, Shape> temp2 = process_image(session_, std::get<0>(temp), std::get<1>(temp));
	auto image = cv::imread(image_path);
	display_image(image, temp2.first, temp2.second);
}
void LoadDnnModel::LoadOnnx()
{
	//model_path 형식맞춰주기
	wchar_t* wPath = new wchar_t[model_path.length() + 1];
	copy(model_path.begin(), model_path.end(), wPath);
	wPath[model_path.length()] = 0;

	session_ = new Ort::Session(env, wPath, Ort::SessionOptions{ nullptr });

	delete[] wPath;


}
pair<Array, Shape> LoadDnnModel::process_image(Ort::Session* session, Array& array, Shape shape)
{
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	input_tensor_ = Ort::Value::CreateTensor<float>(
		memory_info, array.data(), array.size(), shape.data(), shape.size());

	const char* input_names[] = { "images" };
	const char* output_names[] = { "output" };
	auto output = session->Run({}, input_names, &input_tensor_, 1, output_names, 1);
	shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
	auto ptr = output[0].GetTensorData<float>();
	return { Array(ptr, ptr + shape[0] * shape[1]), shape };
}
void LoadDnnModel::display_image(cv::Mat image, const Array& output, const Shape& shape)
{
	// 이미지 리사이즈 비율 계산
	float resize_factor_x = image.cols / 640.0f;
	float resize_factor_y = image.rows / 640.0f;
	for (size_t i = 0; i < shape[0]; ++i)
	{
		auto ptr = output.data() + i * shape[1];
		// 바운딩 박스 좌표와 크기를 이미지 리사이즈 비율에 맞게 조정
		int x = static_cast<int>(ptr[1] * resize_factor_x);
		int y = static_cast<int>(ptr[2] * resize_factor_y);
		int w = static_cast<int>((ptr[3] - ptr[1]) * resize_factor_x);
		int h = static_cast<int>((ptr[4] - ptr[2]) * resize_factor_y);
		int c = ptr[5];
		auto color = CV_RGB(0, 255, 0);
		auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
		cv::rectangle(image, { x, y, w, h }, color);
		cv::putText(image, name, { x, y }, cv::FONT_HERSHEY_DUPLEX, 1, color);
	}
	resultImage = image;
	/*cv::imshow("YOLOv7 Output", image);
	cv::waitKey(0);*/
}
cv::Mat LoadDnnModel::GetResultImage() {
	return resultImage;
}
void LoadDnnModel::drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   // Draw the predicted bounding box
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);

    //Get the label for the class name and its confidence
    string label = cv::format("%.2f", conf);
    label = /*this->class_names[classid]*/"sample";

    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    //rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
    putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 1);
}