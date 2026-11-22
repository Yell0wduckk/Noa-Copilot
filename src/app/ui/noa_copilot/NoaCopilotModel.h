#pragma once

#include "app/ui/noa_copilot/NoaMaterializer.h"
#include "core/chart/model/TimelineData.h"
#include "timeline/TimelineRenderData.h"
#include <QByteArray>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTimer>
#include <QUrl>
#include <QVariantList>
#include <QVariantMap>
#include <memory>

class TimelineQuickModel;
struct SimaiLineCache;

namespace miacode {
class UiRequestService;
}

namespace miacode::ui {

class DocumentModel;
class PreviewModel;
class TimelineModel;

class NoaCopilotModel final : public QObject
{
    Q_OBJECT

    // ---- Core Enable & Status ----
    Q_PROPERTY(bool enabled READ enabled WRITE setEnabled NOTIFY configChanged)
    Q_PROPERTY(QString statusState READ statusState NOTIFY statusChanged)
    Q_PROPERTY(QString daemonState READ statusState NOTIFY statusChanged)
    Q_PROPERTY(QString statusLabel READ statusLabel NOTIFY statusChanged)
    Q_PROPERTY(QString statusMessage READ statusLabel NOTIFY statusChanged)
    Q_PROPERTY(QString statusDetail READ statusDetail NOTIFY statusChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY statusChanged)
    Q_PROPERTY(bool audioAnalyzing READ audioAnalyzing NOTIFY statusChanged)
    Q_PROPERTY(bool daemonReady READ daemonReady NOTIFY statusChanged)
    Q_PROPERTY(QString deviceInfo READ deviceInfo NOTIFY statusChanged)

    // ---- Checkpoints ----
    Q_PROPERTY(bool hasCheckpoint READ hasCheckpoint NOTIFY checkpointsChanged)
    Q_PROPERTY(QString checkpointPath READ checkpointPath WRITE setCheckpointPath NOTIFY checkpointsChanged)
    Q_PROPERTY(QString checkpointDir READ checkpointDir WRITE setCheckpointDir NOTIFY checkpointsChanged)
    Q_PROPERTY(QString activeCheckpointName READ activeCheckpointName NOTIFY checkpointsChanged)
    Q_PROPERTY(QVariantList availableCheckpoints READ availableCheckpoints NOTIFY checkpointsChanged)

    // ---- Primary Generation Controls ----
    Q_PROPERTY(double temperature READ temperature WRITE setTemperature NOTIFY configChanged)
    Q_PROPERTY(QString defaultDesigner READ defaultDesigner WRITE setDefaultDesigner NOTIFY configChanged)
    Q_PROPERTY(QVariantList designerOptions READ designerOptions NOTIFY designerOptionsChanged)
    Q_PROPERTY(QString device READ device WRITE setDevice NOTIFY configChanged)
    Q_PROPERTY(QString deviceMode READ device WRITE setDevice NOTIFY configChanged)
    Q_PROPERTY(QVariantList deviceModeOptions READ deviceModeOptions NOTIFY configChanged)
    Q_PROPERTY(QString precision READ precision WRITE setPrecision NOTIFY configChanged)
    Q_PROPERTY(QString precisionMode READ precision WRITE setPrecision NOTIFY configChanged)
    Q_PROPERTY(QVariantList precisionModeOptions READ precisionModeOptions CONSTANT)

    Q_PROPERTY(bool candidateSessionActive READ candidateSessionActive NOTIFY ghostStateChanged)
    Q_PROPERTY(int continuationLines READ continuationLines WRITE setContinuationLines NOTIFY configChanged)
    Q_PROPERTY(int lineAlignBeats READ lineAlignBeats WRITE setLineAlignBeats NOTIFY configChanged)
    Q_PROPERTY(int candidateCount READ candidateCount WRITE setCandidateCount NOTIFY configChanged)
    Q_PROPERTY(int cpuThreads READ cpuThreads WRITE setCpuThreads NOTIFY configChanged)
    Q_PROPERTY(QString mertDevice READ mertDevice WRITE setMertDevice NOTIFY configChanged)
    Q_PROPERTY(QString cpuPython READ cpuPython WRITE setCpuPython NOTIFY configChanged)
    Q_PROPERTY(QString cudaPython READ cudaPython WRITE setCudaPython NOTIFY configChanged)
    Q_PROPERTY(int selectedCandidate READ selectedCandidate NOTIFY ghostStateChanged)
    Q_PROPERTY(bool skeletonSelection READ skeletonSelection NOTIFY ghostStateChanged)
    Q_PROPERTY(int candidateLocalIssues READ candidateLocalIssues NOTIFY ghostStateChanged)
    Q_PROPERTY(QString transientHint READ transientHint NOTIFY ghostStateChanged)
    Q_PROPERTY(QString environmentStatus READ environmentStatus NOTIFY configChanged)
    Q_PROPERTY(bool cpuMertAvailable READ cpuMertAvailable NOTIFY configChanged)
    Q_PROPERTY(bool cudaMertAvailable READ cudaMertAvailable NOTIFY configChanged)
    Q_PROPERTY(bool cpuAvailable READ cpuAvailable NOTIFY configChanged)
    Q_PROPERTY(bool cudaAvailable READ cudaAvailable NOTIFY configChanged)
    Q_PROPERTY(QVariantList environments READ environments NOTIFY configChanged)
    // ---- Advanced Sampling & Runtime Controls ----
    Q_PROPERTY(double slideProbabilityMultiplier READ slideProbabilityMultiplier WRITE setSlideProbabilityMultiplier NOTIFY configChanged)
    Q_PROPERTY(int seed READ seed WRITE setSeed NOTIFY configChanged)

    Q_PROPERTY(QString pythonMode READ pythonMode WRITE setPythonMode NOTIFY configChanged)
    Q_PROPERTY(QString envMode READ pythonMode WRITE setPythonMode NOTIFY configChanged)
    Q_PROPERTY(QVariantList envModeOptions READ envModeOptions CONSTANT)
    Q_PROPERTY(QString wslPython READ wslPython WRITE setWslPython NOTIFY configChanged)
    Q_PROPERTY(QString wslDistribution READ wslDistribution WRITE setWslDistribution NOTIFY configChanged)
    Q_PROPERTY(QString customPythonPath READ customPythonPath WRITE setCustomPythonPath NOTIFY configChanged)
    Q_PROPERTY(QString resolvedPythonPath READ resolvedPythonPath NOTIFY configChanged)
    Q_PROPERTY(QString sourceMode READ sourceMode WRITE setSourceMode NOTIFY configChanged)
    Q_PROPERTY(QVariantList sourceModeOptions READ sourceModeOptions CONSTANT)
    Q_PROPERTY(QString customSourceRoot READ customSourceRoot WRITE setCustomSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString customSourceDir READ customSourceRoot WRITE setCustomSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString resolvedSourceRoot READ resolvedSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString resolvedSourceDir READ resolvedSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString cacheMessage READ cacheMessage NOTIFY cacheInfoChanged)
    Q_PROPERTY(bool cacheCleaning READ cacheCleaning NOTIFY cacheInfoChanged)
    Q_PROPERTY(QString audioCacheSizeText READ audioCacheSizeText NOTIFY cacheInfoChanged)

    // ---- Workflow 1: Inline Ghost State ----
    Q_PROPERTY(bool ghostActive READ ghostActive NOTIFY ghostStateChanged)
    Q_PROPERTY(bool ghostStreaming READ ghostStreaming NOTIFY ghostStateChanged)
    Q_PROPERTY(QString ghostText READ ghostText NOTIFY ghostStateChanged)
    Q_PROPERTY(QString ghostFirstLine READ ghostFirstLine NOTIFY ghostStateChanged)
    Q_PROPERTY(QString ghostExtraLines READ ghostExtraLines NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostReplaceStart READ ghostReplaceStart NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostReplaceEnd READ ghostReplaceEnd NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostStepCount READ ghostStepCount NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostCurrentStep READ ghostCurrentStep NOTIFY ghostStateChanged)

    // ---- Empty Difficulty Onboarding & Full-Chart Streaming ----
    Q_PROPERTY(bool emptyCardVisible READ emptyCardVisible NOTIFY emptyCardChanged)
    Q_PROPERTY(bool fullStreamingActive READ fullStreamingActive NOTIFY streamingProgressChanged)
    Q_PROPERTY(double streamingProgress READ streamingProgress NOTIFY streamingProgressChanged)
    Q_PROPERTY(int streamingBeat READ streamingBeat NOTIFY streamingProgressChanged)
    Q_PROPERTY(int fullStreamingBeat READ streamingBeat NOTIFY streamingProgressChanged)
    Q_PROPERTY(int streamingTotalBeats READ streamingTotalBeats NOTIFY streamingProgressChanged)
    Q_PROPERTY(int fullStreamingTotalBeats READ streamingTotalBeats NOTIFY streamingProgressChanged)

public:
    explicit NoaCopilotModel(
        DocumentModel& document,
        PreviewModel& preview,
        TimelineModel& timeline,
        miacode::UiRequestService& uiRequests,
        QObject* parent = nullptr);
    ~NoaCopilotModel() override;

    // ---- Getters / Setters ----
    bool enabled() const { return enabled_; }
    void setEnabled(bool value);


    QString statusState() const { return statusState_; }
    QString statusLabel() const { return statusLabel_; }
    QString statusDetail() const { return statusDetail_; }
    bool busy() const { return activeRequestId_ > 0; }
    bool audioAnalyzing() const { return statusState_ == QStringLiteral("warming_audio"); }
    bool daemonReady() const { return daemonConfigured_; }
    QString deviceInfo() const { return deviceInfo_; }

    bool hasCheckpoint() const { return !checkpointPath_.isEmpty(); }
    QString checkpointPath() const { return checkpointPath_; }
    void setCheckpointPath(const QString& path);
    QString checkpointDir() const { return checkpointDir_; }
    void setCheckpointDir(const QString& dir);
    QString activeCheckpointName() const;
    QVariantList availableCheckpoints() const { return availableCheckpoints_; }

    QString defaultDesigner() const { return defaultDesigner_; }
    void setDefaultDesigner(const QString& designer);
    QVariantList designerOptions() const { return designerOptions_; }

    QString device() const { return device_; }
    void setDevice(const QString& value);
    QVariantList deviceModeOptions() const;

    QString precision() const { return precision_; }
    void setPrecision(const QString& value);
    QVariantList precisionModeOptions() const;

    double temperature() const { return temperature_; }
    void setTemperature(double value);
    double slideProbabilityMultiplier() const { return slideProbabilityMultiplier_; }
    void setSlideProbabilityMultiplier(double value);





    int seed() const { return seed_; }
    void setSeed(int value);


    QString pythonMode() const { return pythonMode_; }
    void setPythonMode(const QString& mode);
    QVariantList envModeOptions() const;
    QString wslPython() const { return wslPython_; }
    QString wslDistribution() const { return wslDistribution_; }
    void setWslPython(const QString& path);
    void setWslDistribution(const QString& name);
    QString customPythonPath() const { return customPythonPath_; }
    void setCustomPythonPath(const QString& path);
    QString resolvedPythonPath() const;

    QString sourceMode() const { return sourceMode_; }
    void setSourceMode(const QString& mode);
    QVariantList sourceModeOptions() const;
    QString customSourceRoot() const { return customSourceRoot_; }
    void setCustomSourceRoot(const QString& path);
    QString resolvedSourceRoot() const;

    QString cacheMessage() const { return cacheMessage_; }
    bool cacheCleaning() const { return cacheCleaning_; }
    QString audioCacheSizeText() const { return audioCacheSizeText_; }

    bool ghostActive() const { return ghostActive_; }
    bool ghostStreaming() const { return ghostStreaming_; }
    QString ghostText() const { return ghostText_; }
    QString ghostFirstLine() const { return ghostFirstLine_; }
    QString ghostExtraLines() const { return ghostExtraLines_; }
    int ghostReplaceStart() const { return ghostReplaceStart_; }
    int ghostReplaceEnd() const { return ghostReplaceEnd_; }
    int ghostStepCount() const { return static_cast<int>(ghostSteps_.size()); }
    int ghostCurrentStep() const { return ghostCurrentStep_; }

    bool emptyCardVisible() const;
    bool fullStreamingActive() const { return fullStreamingActive_; }
    double streamingProgress() const { return streamingProgress_; }
    int streamingBeat() const { return streamingBeat_; }
    int streamingTotalBeats() const { return streamingTotalBeats_; }

    bool candidateSessionActive() const;
    int continuationLines() const { return continuationLines_; }
    void setContinuationLines(int value);
    int lineAlignBeats() const { return lineAlignBeats_; }
    void setLineAlignBeats(int value);
    int candidateCount() const { return candidateCount_; }
    void setCandidateCount(int value);
    int cpuThreads() const { return cpuThreads_; }
    void setCpuThreads(int value);
    QString mertDevice() const { return mertDevice_; }
    void setMertDevice(QString value);
    QString cpuPython() const;
    void setCpuPython(QString value);
    QString cudaPython() const { return cudaPython_; }
    void setCudaPython(QString value);
    int selectedCandidate() const { return selected_; }
    bool skeletonSelection() const { return skeletonSelection_; }
    int candidateLocalIssues() const {
        const auto it = candidates_.constFind(selected_);
        return it == candidates_.cend() ? 0 : it->localIssues;
    }
    QString transientHint() const { return exhaustionTimer_.isActive() ? hintText_ : QString(); }
    Q_INVOKABLE bool seekCandidateAt(int visualPosition);
    QString environmentStatus() const { return environmentStatus_; }
    bool cpuMertAvailable() const { return cpuMertAvailable_; }
    bool cudaMertAvailable() const { return cudaMertAvailable_; }
    bool cpuAvailable() const { return cpuAvailable_; }
    bool cudaAvailable() const { return cudaAvailable_; }
    QVariantList environments() const { return environments_; }
    Q_INVOKABLE void scanEnvironments(bool discover = true);
    Q_INVOKABLE void switchCandidate(int direction);
    Q_INVOKABLE void mouseSkeletonSelection(bool quick);
    // ---- Editor & UI Invokables ----
    Q_INVOKABLE void bindEditor(QObject* editorObject);
    Q_INVOKABLE void notifyEditorCursorOrTextChanged(int cursorPos, int selStart, int selEnd, bool textChanged);
    Q_INVOKABLE bool handleCandidateKey(int key);
    Q_INVOKABLE bool handleEditorKeyPress(int key, int modifiers);

    Q_INVOKABLE void triggerManualContinue();
    Q_INVOKABLE bool acceptFullGhost();
    Q_INVOKABLE bool acceptGhostAll() { return acceptFullGhost(); }
    Q_INVOKABLE bool acceptNextGhostStep();
    Q_INVOKABLE bool acceptGhostNextStep() { return acceptNextGhostStep(); }
    Q_INVOKABLE void dismissGhost();
    Q_INVOKABLE void dismissAllActivePreviews();

    Q_INVOKABLE void dismissEmptyCard();
    Q_INVOKABLE void dismissEmptyCardForCurrentChart() { dismissEmptyCard(); }
    Q_INVOKABLE void requestFullChartGeneration(
        const QString& levelOverride = QString(),
        const QString& designerOverride = QString());
    Q_INVOKABLE void triggerFullChartStream() { requestFullChartGeneration(); }
    Q_INVOKABLE void cancelActiveGeneration();

    Q_INVOKABLE void refreshCheckpoints();
    Q_INVOKABLE void openCheckpointFolder();
    Q_INVOKABLE void openCheckpointDirectory() { openCheckpointFolder(); }
    Q_INVOKABLE void chooseCheckpointFileUrl(const QUrl& url);
    Q_INVOKABLE void chooseCustomPythonUrl(const QUrl& url);
    Q_INVOKABLE void chooseCustomSourceDirUrl(const QUrl& url);
    Q_INVOKABLE void browseCheckpointFile();
    Q_INVOKABLE void browseCheckpointDirectory();
    Q_INVOKABLE void browseCustomPythonPath();
    Q_INVOKABLE void browseEnvironmentPython(bool cpu);
    Q_INVOKABLE void browseCustomSourceRoot();
    Q_INVOKABLE void clearAudioCache();
    Q_INVOKABLE void releaseVram();
    Q_INVOKABLE void unloadModelMemory() { releaseVram(); }
    Q_INVOKABLE void restartDaemon();
    Q_INVOKABLE void refreshAudioCacheSize();
    Q_INVOKABLE void setCacheInfoVisible(bool visible) {
        cacheInfoVisible_ = visible;
        if (visible) refreshAudioCacheSize();
    }

signals:
    void configChanged();
    void statusChanged();
    void checkpointsChanged();
    void designerOptionsChanged();
    void cacheInfoChanged();
    void ghostStateChanged();
    void returnedToOriginal();
    void emptyCardChanged();
    void streamingProgressChanged();

private:
    void loadPreferences();
    void savePreferences() const;
    void rebuildDesignerOptions(const QStringList& vocab);
    void updateStatus(const QString& state, const QString& label, const QString& detail = QString());
    void updateEditorGhostStrike();

    QString defaultBuiltinRoot() const;
    QString defaultCheckpointsDir() const;
    void scanCacheIndex(bool clear);
    void finishCacheBarrier();
    bool supportsCacheBarrier_ = false, waitingForCacheBarrier_ = false;
    bool cacheScanActive_ = false, cacheCleaning_ = false;
    bool cacheRefreshPending_ = false, cacheInfoVisible_ = false;
    QList<QJsonObject> deferredCacheRequests_;
    QString cacheMessage_;

    void ensureDaemonRunning();
    void stopDaemon();
    void sendConfigureRequest();
    void sendBindSongIfNeeded();
    void sendSyncChartIfNeeded();
    void sendSuggestRequest(
        const QString& mode,
        const QString& levelOverride,
        const QString& designerOverride);
    void writeDaemonJson(const QJsonObject& payload);
    void handleDaemonReadyReadStdout();
    void handleDaemonReadyReadStderr();
    void handleDaemonMessage(const QJsonObject& message);

    bool publishFullChart(const QString& text);
    void endFullStream();
    void updateGhostSplitLines();
    void applyDualSurfaceGhostPreview(
        const QString& ghostSimai,
        double startSecond,
        double activeBpm,
        int strikeStartUtf16 = 0,
        int strikeEndUtf16 = 0);
    void clearDualSurfaceGhostPreview();
    bool applyEditorReplacement(int startPos, int endPos, const QString& text, int newCursorPos = -1);

    double currentChartBpm() const;
    double currentChartFirstSeconds() const;
    double parseLevelNumeric(const QString& levelText) const;
    QString effectiveDesigner(const QString& overrideDesigner = QString()) const;

    DocumentModel& document_;
    PreviewModel& preview_;
    TimelineModel& timeline_;
    miacode::UiRequestService& uiRequests_;
    QPointer<QObject> editorBridge_;

    int lineAlignBeats_ = 4;
    int continuationLines_ = 4;
    int candidateCount_ = 3;
    int cpuThreads_ = 4;
    QString mertDevice_ = QStringLiteral("cpu");
    QString cpuPython_ = {};
    QString cudaPython_ = {};
    struct Candidate {
        NoaMaterializer writer;
        TimelineAuditionMarkers audition;
        TimelineRenderSnapshot timeline, normalTimeline;
        QString previewKey;
        bool ready = false, previewReady = false;
        int localIssues = 0;
        quint64 analysisId = 0;
        int stepsTaken = 0, left = 0, right = 0, origEnd = 0, cut = 0, bridgeLength = 0;
        QString remainder, remainingPreview;
        double startSecond = 0, activeBpm = 120;
    };
    QMap<int,Candidate> candidates_;
    int selected_ = 0, highCandidate_ = 0, scanGeneration_ = 0;
    bool exhausted_ = false, acceptingStep_ = false, skeletonSelection_ = false;
    bool cpuAvailable_ = false, cudaAvailable_ = false;
    bool cpuMertAvailable_ = false, cudaMertAvailable_ = false;
    QString environmentStatus_;
    quint64 strikeRevision_ = 0, nextAnalysisId_ = 0;
    int strikeLeft_ = -1, strikeRight_ = -1;
    std::pair<int,int> strikeFrom_, strikeTo_;
    QVariantList environments_;
    QJsonObject request_, stepRequest_;
    QTimer renderTimer_, warmTimer_, exhaustionTimer_;
    bool orderedSteps_ = false, needsChartSync_ = true;
    QJsonObject lastWarmSignature_;
    void selectBackendCandidate();
    QString hintText_;
    int prefetchThrough_ = 0;
    void notifyHint(const QString& message);
    std::shared_ptr<SimaiLineCache> ghostLineCache_;
    std::unique_ptr<TimelineQuickModel> ghostTimelineModel_;
    void compileGhost(const QString& text, double start, double bpm, QVector<TimelineNoteMarker>& markers,
                      TimelineRenderSnapshot& timeline);
    void prepareCandidatePreview(Candidate& candidate);
    void beginCandidate(int number, const QJsonObject& layout);
    void showCandidate();
    bool handleStructured(const QJsonObject& message);
    void discardCandidates();
    void requestMore();
    QJsonObject chartRequest() const;
    void warmSong();
    bool selectionHasSkeleton() const;
    // ---- Preferences ----
    bool enabled_ = true;
    QString defaultDesigner_;
    QString device_ = QStringLiteral("auto");
    QString precision_ = QStringLiteral("bf16");
    double temperature_ = 0.5;
    double slideProbabilityMultiplier_ = 1.0;
    double continuationSecond_ = 0.0, continuationEndSecond_ = 0.0;
    int seed_ = -1;
    QString checkpointPath_;
    QString checkpointDir_;
    QString pythonMode_ = QStringLiteral("bundled");
    QString wslPython_;
    QString wslDistribution_;
    bool usesWsl() const;
    QString customPythonPath_;
    QString sourceMode_ = QStringLiteral("bundled");
    QString customSourceRoot_;

    // ---- Runtime & Daemon State ----
    QProcess* daemonProcess_ = nullptr;
    void* windowsJobHandle_ = nullptr;
    QByteArray stdoutBuffer_;
    bool daemonConfigured_ = false;
    bool checkpointLoadedInDaemon_ = false;
    QString statusState_ = QStringLiteral("missing_checkpoint");
    QString statusLabel_ = QStringLiteral("未加载权重");
    QString statusDetail_;
    QString deviceInfo_ = QStringLiteral("CPU / CUDA Auto");
    QVariantList availableCheckpoints_;
    QVariantList designerOptions_;
    QString audioCacheSizeText_ = QStringLiteral("0 MB");

    int nextRpcId_ = 1;
    int activeRequestId_ = 0;
    int pendingBindSongId_ = 0;
    QString activeRequestMode_;
    quint64 activeRequestRevision_ = 0;
    QString boundSongKey_;
    quint64 syncedChartRevision_ = 0;
    QString lastObservedEditorText_;

    QTimer idleVramTimer_;
    int lastCursorPos_ = 0;
    int lastSelStart_ = 0;
    int lastSelEnd_ = 0;

    // ---- Ghost State ----
    bool ghostActive_ = false;
    bool ghostStreaming_ = false;
    bool pendingAcceptAfterStream_ = false;
    QString ghostText_;
    QString ghostFirstLine_;
    QString ghostExtraLines_;
    int ghostReplaceStart_ = 0;
    int ghostReplaceEnd_ = 0;
    int ghostOrigSpanEnd_ = 0;
    int ghostOrigCut_ = 0;
    int ghostBridgeLen_ = 0;
    QJsonArray ghostSteps_;
    int ghostCurrentStep_ = 0;

    // ---- Empty Card & Full Streaming State ----
    int dismissedEmptyDifficultyId_ = -1;
    bool fullStreamingActive_ = false;
    double streamingProgress_ = 0.0;
    int streamingBeat_ = 0;
    int streamingTotalBeats_ = 0;
    QTimer streamFlushTimer_;
    QElapsedTimer lastStreamFlush_, diskFlush_;
    QString pendingStreamChart_;
};

} // namespace miacode::ui
