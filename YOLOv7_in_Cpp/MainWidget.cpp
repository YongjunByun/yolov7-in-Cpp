#include "MainWidget.h"

MainWidget::MainWidget(QWidget* parent)
    : QWidget(parent) {
    setFixedSize(700, 800);

    // �̹����� �� ���¸� ǥ���� QLabel ����
    imageLabel = new QLabel(this);
    imageLabel->setFixedSize(640, 640);
    imageLabel->setStyleSheet("border: 1px solid black");

    modelStatusLabel = new QLabel("Model Not Loaded", this);
    modelStatusLabel->setStyleSheet("QLabel { color : red; }");

    // �̹����� �ҷ����� ���� ��ư�� ���� �ҷ����� ���� ��ư ����
    QPushButton* loadModelButton = new QPushButton("Load Model", this);
    QPushButton* loadImageButton = new QPushButton("Load Image", this);

    // �̹����� �ѱ� �� �ִ� ��ư ����
    QPushButton* nextImageButton = new QPushButton("Next", this);
    QPushButton* prevImageButton = new QPushButton("Previous", this);

    // ��� �̹��� �� ���̾ƿ�
    QVBoxLayout* topLayout = new QVBoxLayout();
    topLayout->addWidget(imageLabel);
    topLayout->setAlignment(imageLabel, Qt::AlignCenter); // �̹��� ���� �߾ӿ� ��ġ��Ŵ

    // ���� ���̾ƿ� ����: �� ���� ������Ʈ
    QVBoxLayout* leftLayout = new QVBoxLayout();
    leftLayout->addWidget(loadModelButton);
    leftLayout->addWidget(modelStatusLabel);

    // ������ ���̾ƿ� ����: �̹��� �ε� �� �׺���̼� ��ư
    QVBoxLayout* rightLayout = new QVBoxLayout();
    rightLayout->addWidget(loadImageButton);
    QHBoxLayout* imageNavigationLayout = new QHBoxLayout();
    imageNavigationLayout->addWidget(prevImageButton);
    imageNavigationLayout->addWidget(nextImageButton);
    rightLayout->addLayout(imageNavigationLayout);

    // �ϴ� �¿� ���̾ƿ��� ���η� ��ġ
    QHBoxLayout* bottomLayout = new QHBoxLayout();
    bottomLayout->addLayout(leftLayout);
    bottomLayout->addLayout(rightLayout);

    // ���� ���� ���̾ƿ��� ��� �� �ϴ� ���̾ƿ� �߰�
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(topLayout);
    mainLayout->addLayout(bottomLayout);
    // ��ư Ŭ�� �ñ׳ΰ� ���� ����
    connect(loadModelButton, &QPushButton::clicked, this, &MainWidget::loadModel);
    connect(loadImageButton, &QPushButton::clicked, this, &MainWidget::loadImage);
    connect(nextImageButton, &QPushButton::clicked, this, &MainWidget::showNextImage);
    connect(prevImageButton, &QPushButton::clicked, this, &MainWidget::showPreviousImage);

    Model = new LoadDnnModel();
}

void MainWidget::loadModel() {
    // �� �ε� ����
    Model->LoadOnnx();
    modelStatusLabel->setText("Model Loaded");
    modelStatusLabel->setStyleSheet("QLabel { color : green; }");
}

void MainWidget::loadImage() {
    // ���� ���̾�α׸� ����Ͽ� �̹��� ���� ����
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");

    if (!fileName.isEmpty()) {
        // ������ �̹��� ������ QLabel�� ǥ��
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
