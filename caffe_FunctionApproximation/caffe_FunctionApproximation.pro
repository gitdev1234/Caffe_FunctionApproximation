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
INCLUDEPATH += include/

LIBS += -lboost_system
LIBS += -lglog
LIBS += -lprotobuf


SOURCES += main.cpp \
    src/ANN.cpp

HEADERS += \
    include/ANN.h



unix:!macx: LIBS += -L$$PWD/../../../../../../CleanMonthly/caffe_repo/caffe/build/lib/ -lcaffe

INCLUDEPATH += $$PWD/../../../../../../CleanMonthly/caffe_repo/caffe/build
DEPENDPATH += $$PWD/../../../../../../CleanMonthly/caffe_repo/caffe/build

unix:!macx: PRE_TARGETDEPS += $$PWD/../../../../../../CleanMonthly/caffe_repo/caffe/build/lib/libcaffe.a
