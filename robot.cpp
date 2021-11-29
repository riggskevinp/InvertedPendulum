/****************************************************************************
**
** Copyright (C) 2016 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "robot.h"

#include <QGraphicsSceneDragDropEvent>
#include <QMimeData>
#include <QPainter>
#include <QParallelAnimationGroup>
#include <QPropertyAnimation>
#include <QtMath>
#include <QRandomGenerator>

#include <iostream>

//! [0]
RobotPart::RobotPart(QGraphicsItem *parent, std::vector<TrajectoryTuple> traj)
    : QGraphicsObject(parent), color(Qt::lightGray), trajectory(traj)
{
    setAcceptDrops(true);
}
//! [0]

//! [1]
void RobotPart::dragEnterEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->mimeData()->hasColor()) {
        event->setAccepted(true);
        dragOver = true;
        update();
    } else {
        event->setAccepted(false);
    }
}

void RobotPart::dragLeaveEvent(QGraphicsSceneDragDropEvent *event)
{
    Q_UNUSED(event);
    dragOver = false;
    update();
}

void RobotPart::dropEvent(QGraphicsSceneDragDropEvent *event)
{
    dragOver = false;
    if (event->mimeData()->hasColor())
        color = qvariant_cast<QColor>(event->mimeData()->colorData());
    update();
}
//! [3]

//! [4]
RobotPendulum::RobotPendulum(QGraphicsItem *parent, std::vector<TrajectoryTuple> traj)
    : RobotPart(parent), trajectory(traj)
{
}

QRectF RobotPendulum::boundingRect() const
{
    return QRectF(7, -200, 13, 200);
}

void RobotPendulum::paint(QPainter *painter,
           const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setBrush(dragOver ? color.lighter(130) : color);
    painter->drawRoundedRect(7, -200, 13, 200, 25, 25, Qt::RelativeSize);

}

void RobotPendulum::dragEnterEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->mimeData()->hasImage()) {
        event->setAccepted(true);
        dragOver = true;
        update();
    } else {
        RobotPart::dragEnterEvent(event);
    }
}

void RobotPendulum::dropEvent(QGraphicsSceneDragDropEvent *event)
{
    if (event->mimeData()->hasImage()) {
        dragOver = false;
        pixmap = qvariant_cast<QPixmap>(event->mimeData()->imageData());
        update();
    } else {
        RobotPart::dropEvent(event);
    }
}

void RobotPendulum::advance(int step){
    if (!step){
        return;
    }
    setTransformOriginPoint(13, 0);

    //qreal rot = ::sin(std::get<1>(trajectory.at(currentIndex++))[2]);
    //std::cout << std::get<2>(trajectory.at(currentIndex++)) << std::endl;
    setRotation(rotation() + std::get<2>(trajectory.at(currentIndex++)));

}

QRectF RobotCart::boundingRect() const
{
    return QRectF(0, -30, 60, 30);
}

void RobotCart::paint(QPainter *painter,
           const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    painter->setBrush(dragOver ? color.lighter(130) : color);
    painter->drawRoundedRect(0, -30, 60, 30, 25, 25, Qt::RelativeSize);
}

void RobotCart::advance(int step){
    if (!step){
        return;
    }
    //std::cout << "cart " << currentIndex << std::endl;
    //std::cout << std::get<1>(trajectory.at(currentIndex++)) << std::endl;
    setPos(QPointF(std::get<1>(trajectory.at(currentIndex++)) * -300,0));
}

//! [10]
Robot::Robot(QGraphicsItem *parent, std::vector<TrajectoryTuple> traj)
    : RobotPart(parent)
{
    setFlag(ItemHasNoContents);

    QGraphicsObject *torsoItem = new RobotCart(this, traj);
    QGraphicsObject *headItem = new RobotPendulum(torsoItem, traj);

    headItem->setPos(14, -30);
}

//! [9]
QRectF Robot::boundingRect() const
{
    return QRectF();
}

void Robot::paint(QPainter *painter,
                  const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(painter);
    Q_UNUSED(option);
    Q_UNUSED(widget);
}
//! [9]
