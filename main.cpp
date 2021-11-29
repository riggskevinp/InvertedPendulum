#include "mainwindow.h"

#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>

#include <QtMath>
#include <QtWidgets>

#include "robot.h"
#include "InvertedPendulumSim.h"

class GraphicsView : public QGraphicsView
{
public:
    using QGraphicsView::QGraphicsView;

protected:
    void resizeEvent(QResizeEvent *) override
    {
    }
};

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);


    auto trajectory = executeSimulation();

    QGraphicsScene scene(-200, -200, 400, 400);
    Robot *robot = new Robot(nullptr, trajectory);
    robot->setTransform(QTransform::fromScale(1.2, 1.2), true);
    robot->setPos(0, 200);
    scene.addItem(robot);



    GraphicsView view(&scene);
    view.setRenderHint(QPainter::Antialiasing);
    view.setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    view.setBackgroundBrush(QColor(230, 200, 167));
    view.setWindowTitle("Inverted Pendulum Robot");
    view.show();

    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, &scene, &QGraphicsScene::advance);
    timer.start(80);

    return a.exec();

}
