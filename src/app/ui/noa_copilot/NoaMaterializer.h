#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QMap>
#include <QString>

namespace miacode::ui {

class NoaMaterializer {
public:
    void begin(const QJsonObject& descriptor, const QString& original, int lineBeats);
    void moment(int tick, const QJsonArray& notes);
    void advance(int tick);
    void finish();
    QString text() const;
    QString preview() const;
    QJsonObject step(int index) const;
    const QJsonObject& layout() const { return layout_; }
    int noteCount() const { return noteCount_; }
    int position() const { return cursor_; }
    int stepCount() const { return cuts_.size() + 1; }
    double bpm(int tick) const;

private:
    struct Cut {
        int offset;
        int division;
        QJsonObject source;
    };
    struct Edit {
        int stop;
        QString text;
    };
    QString skeletonText(int until) const;
    QString rendered(bool preview) const;
    QString interval(int a, int b, int& division, QString body) const;
    QString restore(const QJsonObject& source, double bpm, int division) const;
    QJsonObject layout_;
    QString source_, confirmed_, pending_;
    QMap<int, Edit> replacements_;
    QMap<int, double> tempos_;
    QList<Cut> cuts_;
    QMap<int, QJsonObject> boundaries_;
    int cursor_ = 0, stop_ = 0, origin_ = 0, width_ = 384, division_ = 0, noteCount_ = 0;
    bool full_ = false, skeleton_ = false, done_ = false;
};

} // namespace miacode::ui
