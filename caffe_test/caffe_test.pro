#-------------------------------------------------
#
# Project created by QtCreator 2016-07-30T20:37:11
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = caffe_test
CONFIG   += console
CONFIG   -= app_bundle
CONFIG   += c++11

TEMPLATE = app

DEFINES += CPU_ONLY=1

INCLUDEPATH += /home/anon/Desktop/CleanMonthly/caffe_repo/caffe/include/
INCLUDEPATH += /home/anon/Desktop/CleanMonthly/caffe_repo/caffe/distribute/include/

LIBS += -lboost_system


SOURCES += main.cpp

HEADERS += \
    ann.h



unix:!macx: LIBS += -L$$PWD/../../caffe_repo/caffe/build/lib/ -lcaffe

INCLUDEPATH += $$PWD/../../caffe_repo/caffe/build
DEPENDPATH += $$PWD/../../caffe_repo/caffe/build
