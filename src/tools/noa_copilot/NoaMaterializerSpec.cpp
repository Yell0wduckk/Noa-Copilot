#include "app/ui/noa_copilot/NoaMaterializer.h"
#include "core/analysis/TimelineSlowRefresh.h"

#include "core/scene/PreviewSfxTimeline.h"
#include "common/GrayscaleImage.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <cmath>

using miacode::ui::NoaAuditionRange;
using miacode::ui::NoaMaterializer;
using miacode::ui::noaAuditionMarkers;

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

bool sameSound(const QVector<TimelineNoteMarker>& heard, const QString& expected)
{
    using namespace miacode::preview_sfx_timeline;
    const auto cold = buildTimelinePreviewRefreshState(SimaiParser::parseForTimeline(expected), 0.125);
    QVector<Event> actual, reference;
    QVector<TouchholdSpan> loops, expectedLoops;
    buildTimeline(heard, 1.0, {}, &actual, &loops);
    buildTimeline(cold.shiftedNoteMarkers, 1.0, {}, &reference, &expectedLoops);
    if (actual.size() != reference.size() || loops.size() != expectedLoops.size()) return false;
    for (int i = 0; i < actual.size(); ++i) {
        const auto& a = actual[i];
        const auto& b = reference[i];
        if (a.kind != b.kind || std::abs(a.second - b.second) > 1e-8 || a.priority != b.priority
            || a.gain != b.gain || a.spanIndex != b.spanIndex || a.breakSlideTailBreak != b.breakSlideTailBreak) return false;
    }
    for (int i = 0; i < loops.size(); ++i)
        if (std::abs(loops[i].startSecond - expectedLoops[i].startSecond) > 1e-8
            || std::abs(loops[i].endSecond - expectedLoops[i].endSecond) > 1e-8) return false;
    return true;
}

bool auditionSequence()
{
    const QString prefix = "(120){4}1-5[2:1]/Ch[2:1],\n{4}";
    const QString suffix = "\n(180){8}8b,7,\nE";
    const QString original = prefix + "4b,5-1[4:1]," + suffix;
    const int left = prefix.size(), right = original.size() - suffix.size();
    const QJsonObject layout{{"start",96},{"stop",288},{"left",left},{"right",right},{"division",4},
        {"initial_bpm",120},{"origin",0},{"tempos",QJsonArray{QJsonArray{0,120},QJsonArray{288,180}}},
        {"cuts",QJsonArray{QJsonObject{{"tick",192},{"position",left+3},{"next_tick",192},{"second",1.0}}}}};
    NoaMaterializer writer;
    std::shared_ptr<SimaiLineCache> cache;
    auto check = [&](QString expected, int ghosts) {
        const auto text = writer.auditionText();
        const NoaAuditionRange range{{2,4},{2,4+int(text.size())}};
        const auto heard = noaAuditionMarkers(prefix + text + suffix,range,.125,{},cache);
        return heard.ghost.size() == ghosts && heard.normal.size() == 4 &&
            sameSound(heard.playback, prefix+expected+suffix);
    };
    auto tap = [](int pad,int flags) { return QJsonArray{QJsonObject{{"pad",QString::number(pad)},{"flags",flags}}}; };
    writer.begin(layout,original,4);
    if (!check(",,",0)) return false;
    writer.moment(96,tap(3,1));
    if (!check("3b,,",1)) return false;
    writer.moment(192,tap(4,2));
    writer.finish();
    if (!check("3b,4x,",2) || writer.text()!=writer.auditionText()) return false;
    const auto step = writer.step(0);
    const QString committed = prefix + step["accept_text"].toString() + step["escape_bridge"].toString()
        + original.mid(step["replace_end_u16"].toInt());
    const auto restored = noaAuditionMarkers(committed, {}, .125, {}, cache);
    if (!sameSound(restored.playback,prefix+"3b,5-1[4:1],"+suffix)) return false;
    writer.begin(layout,original,4);
    if (!check(",,",0)) return false;
    writer.moment(192,tap(6,0)); writer.finish();
    if (!check(",6,",1)) return false;
    return sameSound(noaAuditionMarkers(original, {}, .125, {}, cache).playback, original);
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
    using miacode::ui::noaWarmTimingText;
    const QString withNotes = "(150){8}1,2,|| (90),\n{#0.12}3,(180)4,{6}5-2[4:1],";
    const QString edited = "(150){8}2-4[4:1]/8,x,|| (110),,\n{#0.12}7,(180)*,{16}8,";
    if (noaWarmTimingText(withNotes) != noaWarmTimingText(edited)
        || noaWarmTimingText(edited) != QStringLiteral("(150){8},,{#0.12},(180),")
        || noaWarmTimingText(QStringLiteral("(150)1,,(180)x,")) == noaWarmTimingText(QStringLiteral("(150)1,(180)x,"))
        || !noaWarmTimingText(QStringLiteral("1,2,|| (100)\nE")).isEmpty()) return 1;
    if (!auditionSequence()) { qWarning()<<"audition sound-table mismatch"; return 1; }
    QImage image(1,1,QImage::Format_ARGB32);
    image.setPixelColor(0,0,QColor(240,240,240,128));
    if (miacode::grayscaleImage(image).pixelColor(0,0) != QColor(180,180,180,128)) return 1;
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
