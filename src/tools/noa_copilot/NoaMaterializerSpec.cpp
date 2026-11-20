#include "app/ui/noa_copilot/NoaMaterializer.h"
#include "core/analysis/TimelineSlowRefresh.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <cmath>

using miacode::ui::NoaMaterializer;

namespace {
bool compareParses(const QString& text, std::shared_ptr<SimaiLineCache>& cache)
{
    const auto incremental = SimaiParser::parseIncremental(text, {}, cache);
    const auto cold = SimaiParser::parseForTimeline(text);
    const auto a = buildTimelinePreviewRefreshState(incremental, 0);
    const auto b = buildTimelinePreviewRefreshState(cold, 0);
    if (a.noteMarkerSignature != b.noteMarkerSignature || incremental.ok != cold.ok
        || incremental.errors.size() != cold.errors.size() || incremental.warnings.size() != cold.warnings.size()
        || incremental.beatMarkers.size() != cold.beatMarkers.size() || incremental.measureLineSeconds.size() != cold.measureLineSeconds.size()
        || std::abs(incremental.durationSeconds - cold.durationSeconds) > 1e-8) return false;
    for (qsizetype i = 0; i < cold.beatMarkers.size(); ++i) {
        const auto& x = incremental.beatMarkers[i];
        const auto& y = cold.beatMarkers[i];
        if (std::abs(x.second-y.second)>1e-8 || x.major!=y.major || x.sourceLine!=y.sourceLine || x.sourceCol!=y.sourceCol) return false;
    }
    for (qsizetype i = 0; i < cold.measureLineSeconds.size(); ++i)
        if (std::abs(incremental.measureLineSeconds[i]-cold.measureLineSeconds[i])>1e-8) return false;
    return true;
}

bool replay(const QJsonObject& fixture)
{
    const QString source = fixture["text"].toString();
    NoaMaterializer writer;
    int expected = 0, left = 0, right = 0, completed = 0;
    for (auto entry : fixture["events"].toArray()) {
        const auto m = entry.toObject();
        const auto event = m["event"].toString();
        if (event == "candidate_begin") {
            writer.begin(m,source,4);
            left = m["left"].toInt();
            right = m["right"].toInt();
            expected=0;
        } else if (event == "moment") {
            const auto notes=m["notes"].toArray();
            expected+=notes.size();
            writer.moment(m["tick"].toInt(),notes);
        } else if (event=="beat") {
            writer.advance(m["tick"].toInt());
        } else if (event=="candidate_end") {
            writer.finish();
            const QString chart=source.left(left)+writer.text()+source.mid(right);
            const auto parsed=SimaiParser::parseForTimeline(chart);
            const auto ghost=SimaiParser::parseForTimeline(QStringLiteral("(%1)").arg(writer.bpm(writer.layout()["start"].toInt()))+writer.preview()+"\nE");
            if (!parsed.ok || !ghost.ok || writer.noteCount()!=expected || ghost.noteMarkers.size()<expected) {
                qWarning() << "materialized candidate" << m["candidate"] << chart << "expected" << expected << "markers" << ghost.noteMarkers.size();
                for (const auto& error : parsed.errors) qWarning()<<error.message;
                return false;
            }
            ++completed;
        }
    }
    return completed>0;
}
}

int main(int argc,char** argv)
{
    QCoreApplication app(argc,argv);
    const QString start="(120){4}1h[1:1],x,*,4,\n";
    const QString suffix="{8}2-5[4:1],3,4,5,\n(180){4}6,x**,8,\nE";
    std::shared_ptr<SimaiLineCache> cache;
    for (const auto& text : {start+suffix, start+"{8}1,2,\n"+suffix, start+"{4}3,\n"+suffix,
                              start+suffix, start+"(160){12}6b,7,8,\n"+suffix, start+"xyz,\n"+suffix}) {
        if (!compareParses(text,cache)) { qWarning()<<"incremental mismatch"<<text; return 1; }
    }
    const auto skeleton=SimaiParser::parseForTimeline("(120){8}x,x*,x**,1x,1-5[4:1],\nE");
    if (!skeleton.ok || skeleton.noteMarkers.size()<5 ||
        std::count_if(skeleton.noteMarkers.cbegin(),skeleton.noteMarkers.cend(),[](const auto& m){ return m.skeleton && m.lane==1; })!=3) {
        qWarning()<<"placeholder recognition"; return 1;
    }
    NoaMaterializer writer;
    writer.begin({{"mode","full"},{"start",0},{"stop",389},{"tempos",QJsonArray{QJsonArray{0,120},QJsonArray{192,180}}}}, {}, 4);
    writer.moment(0,QJsonArray{QJsonObject{{"pad","1"},{"group",0},{"kind",1},{"end",96},{"seconds",.5}}});
    writer.advance(96); writer.advance(192); writer.advance(288); writer.advance(384); writer.finish();
    if (writer.text()!="(120){4}1h[4:1],,(180),,\n{4},\nE") { qWarning()<<writer.text(); return 1; }
    for (const QString& path : app.arguments().mid(1)) {
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly) || !replay(QJsonDocument::fromJson(f.readAll()).object())) return 1;
    }
    qInfo()<<"Noa materialization and incremental parser passed";
}
