// Contract regression for the Preview half of the Preview/Timeline split.
//
// PreviewHost owns the PreviewSurface projection boundary, but transport
// commands and the authoritative chart time must travel through the playback
// port. The legacy PreviewSurface remains only the transitional render/settings
// implementation in this stage.

#include "app/runtime/preview/PreviewHost.h"

#include <QCoreApplication>
#include <QStringList>
#include <QTextStream>

namespace {

bool require(bool condition, const QString& message, QTextStream& err)
{
    if (!condition) {
        err << "FAIL: " << message << Qt::endl;
    }
    return condition;
}

class FakePreviewSurface final : public miacode::PreviewSurface
{
public:
    bool playing() const override { return playingValue; }
    miacode::PlaybackTransportState playbackTransportState() const override
    {
        return transportState;
    }
    double positionSeconds() const override { return position; }
    double durationSeconds() const override { return duration; }
    double lowerBoundSeconds() const override { return lowerBound; }
    double playbackRate() const override { return rate; }
    QString playbackRateLabel() const override { return QStringLiteral("legacy-label"); }
    QObject* previewRuntimeObject() const override { return runtimeObject; }
    QObject* stageMediaHostObject() const override { return stageMediaObject; }
    double canvasAspectRatio() const override { return 1.5; }
    QStringList statsTexts() const override { return {QStringLiteral("stats")}; }
    RenderMode muriRenderMode() const override { return RenderMode::Native; }
    void setMuriRenderMode(RenderMode) override { ++setMuriModeCount; }
    void toggleMuriRenderMode() override { ++toggleMuriModeCount; }
    int muriHandRadiusPx() const override { return 30; }
    void setMuriHandRadiusPx(int) override {}
    int muriTapOnSlideThresholdMs() const override { return 200; }
    void setMuriTapOnSlideThresholdMs(int) override {}
    QStringList availableSkinDirectoryNames() const override { return {}; }
    QString skinDisplayName(const QString&) const override { return {}; }
    QString resolveSkinDir() const override { return QStringLiteral("skin"); }
    QString resolveSkinRootDir() const override { return QStringLiteral("root"); }
    QString resolveCustomOutlineDir() const override { return QStringLiteral("outline"); }
    QStringList availableCustomOutlineFileNames() const override { return {}; }
    QString customOutlineDisplayName(const QString&) const override { return {}; }
    QString currentCustomOutlineFileName() const override { return {}; }
    void applyOutlineVariant(PreviewOutlineVariant, bool, bool) override {}
    void applyCustomOutlineFileName(const QString&, bool) override {}
    QVariantMap renderSettings() const override { return settings; }
    void setRenderSetting(const QString&, const QVariant&) override { ++setRenderSettingCount; }
    void refreshSurfaces() override { ++refreshSurfacesCount; }
    void applySfxLevels() override { ++applySfxCount; }
    void prepareForShutdown() override { ++shutdownCount; }
    PreviewAudioSettings audioSettings() const override { return {}; }
    void applyAudioSettings(const PreviewAudioSettings&) override { ++applyAudioSettingsCount; }
    void saveAudioSettingsAsSoftwareDefault() override { ++saveAudioSettingsCount; }
    void restoreAudioSettingsFromSoftwareDefault() override { ++restoreAudioSettingsCount; }
    void setCopilotReview(bool active) override { reviewing = active; }
    void setGhostPreviewOverlay(const QVector<TimelineNoteMarker>& notes, bool strike,
                                int firstLine, int firstColumn, int lastLine, int lastColumn,
                                bool streaming, const TimelineAuditionMarkers* audition) override
    {
        ghostNotes = notes;
        strikeRange = strike ? QList<int>{firstLine, firstColumn, lastLine, lastColumn} : QList<int>{};
        receivingStream = streaming;
        receivedAudition = audition;
    }
    QFuture<int> localGhostIssues(const QVector<TimelineNoteMarker>& notes, double start, double end) override
    {
        int count = 0;
        for (const auto& note : notes) {
            if (start <= note.second && note.second < end) ++count;
        }
        return QtFuture::makeReadyValueFuture(count);
    }
    void clearGhostPreviewOverlay(bool accepted) override
    {
        clearDecisions.append(accepted);
        ghostNotes.clear();
        receivedAudition = nullptr;
    }

    bool reviewing = false;
    bool receivingStream = false;
    QList<int> strikeRange;
    QVector<TimelineNoteMarker> ghostNotes;
    const TimelineAuditionMarkers* receivedAudition = nullptr;
    QList<bool> clearDecisions;
    bool playingValue = false;
    miacode::PlaybackTransportState transportState =
        miacode::PlaybackTransportState::Paused;
    double position = 2.0;
    double duration = 99.0;
    double lowerBound = -3.0;
    double rate = 1.25;
    QObject* runtimeObject = nullptr;
    QObject* stageMediaObject = nullptr;
    QVariantMap settings{{QStringLiteral("quality"), QStringLiteral("high")}};
    int setMuriModeCount = 0;
    int toggleMuriModeCount = 0;
    int setRenderSettingCount = 0;
    int refreshSurfacesCount = 0;
    int applySfxCount = 0;
    int shutdownCount = 0;
    int applyAudioSettingsCount = 0;
    int saveAudioSettingsCount = 0;
    int restoreAudioSettingsCount = 0;
};

class FakePreviewPlaybackPort final : public miacode::PreviewPlaybackPort,
                                      public miacode::AudioClockSource
{
public:
    miacode::PlaybackSnapshot playbackSnapshot() const override { return snapshot; }
    bool acceptsPlaybackCallback(const miacode::PlaybackCallbackStamp& stamp) const override
    {
        return stamp == snapshot.stamp();
    }
    double currentAudioClockSecond() const override { return snapshot.canonicalChartTime; }
    void togglePlayback() override { ++toggleCount; }
    void stop() override { ++stopCount; }
    void seek(double second) override
    {
        snapshot.canonicalChartTime = second;
        ++seekCount;
    }
    void beginScrub() override { ++beginScrubCount; }
    void updateScrub(double second) override
    {
        snapshot.canonicalChartTime = second;
        ++updateScrubCount;
    }
    void endScrub(double second) override
    {
        snapshot.canonicalChartTime = second;
        ++endScrubCount;
    }
    void setPlaybackRate(double rate) override
    {
        snapshot.playbackRate = rate;
        ++setRateCount;
    }
    void nudgePlaybackRate(int) override { ++nudgeRateCount; }

    miacode::PlaybackSnapshot snapshot{
        5, 13, 8, 17.5, 120.0, -4.0, 1.75,
        miacode::PlaybackTransportState::Playing};
    int toggleCount = 0;
    int stopCount = 0;
    int seekCount = 0;
    int beginScrubCount = 0;
    int updateScrubCount = 0;
    int endScrubCount = 0;
    int setRateCount = 0;
    int nudgeRateCount = 0;
};

bool verifyPreviewUsesPlaybackPort(QTextStream& err)
{
    FakePreviewSurface legacy;
    FakePreviewPlaybackPort port;
    miacode::runtime::PreviewHost host(legacy, port, port);

    const auto snapshot = host.playbackSnapshot();
    bool ok = require(host.playing() && host.playbackTransportState()
                          == miacode::PlaybackTransportState::Playing
                          && qFuzzyCompare(host.positionSeconds(), 17.5)
                          && qFuzzyCompare(host.durationSeconds(), 120.0)
                          && qFuzzyCompare(host.lowerBoundSeconds(), -4.0)
                          && qFuzzyCompare(host.playbackRate(), 1.75)
                          && snapshot.sessionGeneration == 5,
                      QStringLiteral("preview transport reads the stamped playback port"), err);
    return ok;
}

bool verifyPreviewProjectionAndInvalidation(QTextStream& err)
{
    FakePreviewSurface legacy;
    FakePreviewPlaybackPort port;
    miacode::runtime::PreviewHost host(legacy, port, port);
    bool ok = require(host.previewRuntimeObject() == nullptr
                          && host.stageMediaHostObject() == nullptr
                          && qFuzzyCompare(host.canvasAspectRatio(), 1.5)
                          && host.statsTexts() == QStringList{QStringLiteral("stats")}
                          && host.renderSettings() == legacy.settings,
                      QStringLiteral("preview resource and settings projections stay in the legacy host"), err);
    host.refreshSurfaces();
    host.applySfxLevels();
    host.prepareForShutdown();
    ok &= require(legacy.refreshSurfacesCount == 1 && legacy.applySfxCount == 1
                      && legacy.shutdownCount == 1,
                  QStringLiteral("non-transport preview operations remain delegated"), err);
    host.invalidateSession();
    const int refreshCountBeforeWithdrawnCommand = legacy.refreshSurfacesCount;
    host.refreshSurfaces();
    ok &= require(!host.playing() && host.playbackSnapshot().transportState
                          == miacode::PlaybackTransportState::Stopped
                      && legacy.refreshSurfacesCount == refreshCountBeforeWithdrawnCommand,
                  QStringLiteral("an invalidated preview host is inert"), err);
    return ok;
}

bool verifyCopilotAudition(QTextStream& err)
{
    FakePreviewSurface playback;
    FakePreviewPlaybackPort port;
    miacode::runtime::PreviewHost host(playback, port, port);
    miacode::PreviewSurface& surface = host;
    TimelineNoteMarker note;
    note.second = 2;
    note.lane = 1;
    TimelineAuditionMarkers first;
    first.ghost.append(note);
    first.playback = first.ghost;
    surface.setCopilotReview(true);
    surface.setGhostPreviewOverlay(first.ghost, true, 3, 4, 5, 6, true, &first);
    bool ok = require(playback.reviewing && playback.receivingStream && playback.receivedAudition == &first
                          && playback.ghostNotes.isSharedWith(first.ghost)
                          && playback.strikeRange == QList<int>{3, 4, 5, 6},
                      QStringLiteral("streaming candidate reaches the workbench preview surface"), err);

    TimelineAuditionMarkers second;
    note.lane = 6;
    second.ghost.append(note);
    second.playback = second.ghost;
    surface.setGhostPreviewOverlay(second.ghost, false, 0, 0, 0, 0, false, &second);
    ok &= require(!playback.receivingStream && playback.strikeRange.isEmpty()
                      && playback.ghostNotes.isSharedWith(second.ghost) && playback.receivedAudition == &second
                      && surface.localGhostIssues(second.ghost, 1.5, 2.5).result() == 1,
                  QStringLiteral("switching candidates replaces both visual and audible markers"), err);
    surface.clearGhostPreviewOverlay(false);
    surface.setGhostPreviewOverlay(second.ghost, false, 0, 0, 0, 0, false, &second);
    surface.clearGhostPreviewOverlay(true);
    surface.setCopilotReview(false);
    ok &= require(!playback.reviewing && playback.ghostNotes.isEmpty() && !playback.receivedAudition
                      && playback.clearDecisions == QList<bool>{false, true},
                  QStringLiteral("discard and accept reach playback with the correct decision"), err);
    return ok;
}

}  // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    QTextStream err(stderr);
    bool ok = true;
    ok &= verifyPreviewUsesPlaybackPort(err);
    ok &= verifyPreviewProjectionAndInvalidation(err);
    ok &= verifyCopilotAudition(err);
    if (ok) {
        QTextStream(stdout) << "preview_host_spec: OK" << Qt::endl;
    }
    return ok ? 0 : 1;
}
