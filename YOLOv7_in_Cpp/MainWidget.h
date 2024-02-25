#pragma once

#include <QtWidgets/QWidget>
#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QPixmap>
//#include "ui_MainWidget.h"

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget* parent = nullptr);

private slots:
    void loadImage();
    void loadModel();
    void showNextImage();
    void showPreviousImage();
private:
    QLabel* imageLabel;
    QLabel* modelStatusLabel;
};