#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "LoadDnnModel.h"

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/qt/qml/yolov7_in_cpp/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    LoadDnnModel sample;//test
    sample.LoadOnnx();

    return app.exec();
}
