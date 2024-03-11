#include "MainWidget.h"

MainWidget::MainWidget(QWidget* parent)
    : QWidget(parent) {
    setFixedSize(700, 800);

    // 이미지와 모델 상태를 표시할 QLabel 생성
    imageLabel = new QLabel(this);
    imageLabel->setFixedSize(640, 640);
    imageLabel->setStyleSheet("border: 1px solid black");

    modelStatusLabel = new QLabel("Model Not Loaded", this);
    modelStatusLabel->setStyleSheet("QLabel { color : red; }");

    // 이미지를 불러오기 위한 버튼과 모델을 불러오기 위한 버튼 생성
    QPushButton* loadModelButton = new QPushButton("Load Model", this);
    QPushButton* loadImageButton = new QPushButton("Load Image", this);

    // 이미지를 넘길 수 있는 버튼 생성
    QPushButton* nextImageButton = new QPushButton("Next", this);
    QPushButton* prevImageButton = new QPushButton("Previous", this);

    // 상단 이미지 라벨 레이아웃
    QVBoxLayout* topLayout = new QVBoxLayout();
    topLayout->addWidget(imageLabel);
    topLayout->setAlignment(imageLabel, Qt::AlignCenter); // 이미지 라벨을 중앙에 위치시킴

    // 왼쪽 레이아웃 구성: 모델 관련 컴포넌트
    QVBoxLayout* leftLayout = new QVBoxLayout();
    leftLayout->addWidget(loadModelButton);
    leftLayout->addWidget(modelStatusLabel);

    // 오른쪽 레이아웃 구성: 이미지 로딩 및 네비게이션 버튼
    QVBoxLayout* rightLayout = new QVBoxLayout();
    rightLayout->addWidget(loadImageButton);
    QHBoxLayout* imageNavigationLayout = new QHBoxLayout();
    imageNavigationLayout->addWidget(prevImageButton);
    imageNavigationLayout->addWidget(nextImageButton);
    rightLayout->addLayout(imageNavigationLayout);

    // 하단 좌우 레이아웃을 가로로 배치
    QHBoxLayout* bottomLayout = new QHBoxLayout();
    bottomLayout->addLayout(leftLayout);
    bottomLayout->addLayout(rightLayout);

    // 최종 메인 레이아웃에 상단 및 하단 레이아웃 추가
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(topLayout);
    mainLayout->addLayout(bottomLayout);
    // 버튼 클릭 시그널과 슬롯 연결
    connect(loadModelButton, &QPushButton::clicked, this, &MainWidget::loadModel);
    connect(loadImageButton, &QPushButton::clicked, this, &MainWidget::loadImage);
    connect(nextImageButton, &QPushButton::clicked, this, &MainWidget::showNextImage);
    connect(prevImageButton, &QPushButton::clicked, this, &MainWidget::showPreviousImage);

    Model = new LoadDnnModel();
}

void MainWidget::loadModel() {
    // 모델 로딩 로직
    Model->LoadOnnx();
    modelStatusLabel->setText("Model Loaded");
    modelStatusLabel->setStyleSheet("QLabel { color : green; }");
}

void MainWidget::loadImage() {
    // 파일 다이얼로그를 사용하여 이미지 파일 선택
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");

    if (!fileName.isEmpty()) {
        // 선택한 이미지 파일을 QLabel에 표시
        string imgpath = fileName.toStdString();
        Model->SetImgPath(imgpath);
        Model->Run();
        cv::Mat resultImage = Model->GetResultImage();
        QImage qImage(resultImage.data, resultImage.cols, resultImage.rows, int(resultImage.step), QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(qImage);
        imageLabel->setPixmap(pixmap.scaled(640, 640, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
}

void MainWidget::showNextImage() {
}
void MainWidget::showPreviousImage() {
}
