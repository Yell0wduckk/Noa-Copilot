#pragma once

#include <QByteArray>
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
    Q_PROPERTY(bool autoContinue READ autoContinue WRITE setAutoContinue NOTIFY configChanged)
    Q_PROPERTY(bool autoContinueEnabled READ autoContinue WRITE setAutoContinue NOTIFY configChanged)
    Q_PROPERTY(QString statusState READ statusState NOTIFY statusChanged)
    Q_PROPERTY(QString daemonState READ statusState NOTIFY statusChanged)
    Q_PROPERTY(QString statusLabel READ statusLabel NOTIFY statusChanged)
    Q_PROPERTY(QString statusMessage READ statusLabel NOTIFY statusChanged)
    Q_PROPERTY(QString statusDetail READ statusDetail NOTIFY statusChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY statusChanged)
    Q_PROPERTY(bool daemonReady READ daemonReady NOTIFY statusChanged)
    Q_PROPERTY(QString deviceInfo READ deviceInfo NOTIFY statusChanged)

    // ---- Checkpoints ----
    Q_PROPERTY(bool hasCheckpoint READ hasCheckpoint NOTIFY checkpointsChanged)
    Q_PROPERTY(QString checkpointPath READ checkpointPath WRITE setCheckpointPath NOTIFY checkpointsChanged)
    Q_PROPERTY(QString checkpointDir READ checkpointDir WRITE setCheckpointDir NOTIFY checkpointsChanged)
    Q_PROPERTY(QString activeCheckpointName READ activeCheckpointName NOTIFY checkpointsChanged)
    Q_PROPERTY(QVariantList availableCheckpoints READ availableCheckpoints NOTIFY checkpointsChanged)

    // ---- Primary Generation Controls ----
    Q_PROPERTY(QString creativityPreset READ creativityPreset WRITE setCreativityPreset NOTIFY configChanged)
    Q_PROPERTY(QVariantList creativityPresetOptions READ creativityPresetOptions CONSTANT)
    Q_PROPERTY(QString defaultDesigner READ defaultDesigner WRITE setDefaultDesigner NOTIFY configChanged)
    Q_PROPERTY(QVariantList designerOptions READ designerOptions NOTIFY designerOptionsChanged)
    Q_PROPERTY(QString device READ device WRITE setDevice NOTIFY configChanged)
    Q_PROPERTY(QString deviceMode READ device WRITE setDevice NOTIFY configChanged)
    Q_PROPERTY(QVariantList deviceModeOptions READ deviceModeOptions CONSTANT)
    Q_PROPERTY(QString precision READ precision WRITE setPrecision NOTIFY configChanged)
    Q_PROPERTY(QString precisionMode READ precision WRITE setPrecision NOTIFY configChanged)
    Q_PROPERTY(QVariantList precisionModeOptions READ precisionModeOptions CONSTANT)

    // ---- Advanced Sampling & Runtime Controls ----
    Q_PROPERTY(double temperature READ temperature WRITE setTemperature NOTIFY configChanged)
    Q_PROPERTY(int topK READ topK WRITE setTopK NOTIFY configChanged)
    Q_PROPERTY(double topP READ topP WRITE setTopP NOTIFY configChanged)
    Q_PROPERTY(double repetitionPenalty READ repetitionPenalty WRITE setRepetitionPenalty NOTIFY configChanged)
    Q_PROPERTY(int maxSlideJudgeAreas READ maxSlideJudgeAreas WRITE setMaxSlideJudgeAreas NOTIFY configChanged)
    Q_PROPERTY(int seed READ seed WRITE setSeed NOTIFY configChanged)
    Q_PROPERTY(int debounceMs READ debounceMs WRITE setDebounceMs NOTIFY configChanged)

    Q_PROPERTY(QString pythonMode READ pythonMode WRITE setPythonMode NOTIFY configChanged)
    Q_PROPERTY(QString envMode READ pythonMode WRITE setPythonMode NOTIFY configChanged)
    Q_PROPERTY(QVariantList envModeOptions READ envModeOptions CONSTANT)
    Q_PROPERTY(QString customPythonPath READ customPythonPath WRITE setCustomPythonPath NOTIFY configChanged)
    Q_PROPERTY(QString resolvedPythonPath READ resolvedPythonPath NOTIFY configChanged)
    Q_PROPERTY(QString sourceMode READ sourceMode WRITE setSourceMode NOTIFY configChanged)
    Q_PROPERTY(QVariantList sourceModeOptions READ sourceModeOptions CONSTANT)
    Q_PROPERTY(QString customSourceRoot READ customSourceRoot WRITE setCustomSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString customSourceDir READ customSourceRoot WRITE setCustomSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString resolvedSourceRoot READ resolvedSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString resolvedSourceDir READ resolvedSourceRoot NOTIFY configChanged)
    Q_PROPERTY(QString audioCacheSizeText READ audioCacheSizeText NOTIFY cacheInfoChanged)

    // ---- Workflow 1: Inline Ghost State ----
    Q_PROPERTY(bool ghostActive READ ghostActive NOTIFY ghostStateChanged)
    Q_PROPERTY(QString ghostText READ ghostText NOTIFY ghostStateChanged)
    Q_PROPERTY(QString ghostFirstLine READ ghostFirstLine NOTIFY ghostStateChanged)
    Q_PROPERTY(QString ghostExtraLines READ ghostExtraLines NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostReplaceStart READ ghostReplaceStart NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostReplaceEnd READ ghostReplaceEnd NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostStepCount READ ghostStepCount NOTIFY ghostStateChanged)
    Q_PROPERTY(int ghostCurrentStep READ ghostCurrentStep NOTIFY ghostStateChanged)

    // ---- Workflow 2: Inline Range Card (Ctrl+K) & Pending Review ----
    Q_PROPERTY(bool inlineCardVisible READ inlineCardVisible WRITE setInlineCardVisible NOTIFY inlineCardChanged)
    Q_PROPERTY(int inlineSpanBeats READ inlineSpanBeats WRITE setInlineSpanBeats NOTIFY inlineCardChanged)
    Q_PROPERTY(bool inlineFillToNext READ inlineFillToNext WRITE setInlineFillToNext NOTIFY inlineCardChanged)
    Q_PROPERTY(QString inlineLevelOverride READ inlineLevelOverride WRITE setInlineLevelOverride NOTIFY inlineCardChanged)
    Q_PROPERTY(QString inlineDesignerOverride READ inlineDesignerOverride WRITE setInlineDesignerOverride NOTIFY inlineCardChanged)
    Q_PROPERTY(bool pendingRangeActive READ pendingRangeActive NOTIFY pendingRangeChanged)
    Q_PROPERTY(QString pendingRangeText READ pendingRangeText NOTIFY pendingRangeChanged)
    Q_PROPERTY(int pendingReplaceStart READ pendingReplaceStart NOTIFY pendingRangeChanged)
    Q_PROPERTY(int pendingReplaceEnd READ pendingReplaceEnd NOTIFY pendingRangeChanged)
    Q_PROPERTY(double pendingStartSecond READ pendingStartSecond NOTIFY pendingRangeChanged)
    Q_PROPERTY(double pendingEndSecond READ pendingEndSecond NOTIFY pendingRangeChanged)
    Q_PROPERTY(int pendingNoteCount READ pendingNoteCount NOTIFY pendingRangeChanged)
    Q_PROPERTY(int pendingRangeNoteCount READ pendingNoteCount NOTIFY pendingRangeChanged)

    // ---- Workflow 3: Empty Difficulty Onboarding & Full-Chart Streaming ----
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

    bool autoContinue() const { return autoContinue_; }
    void setAutoContinue(bool value);

    QString statusState() const { return statusState_; }
    QString statusLabel() const { return statusLabel_; }
    QString statusDetail() const { return statusDetail_; }
    bool busy() const { return activeRequestId_ > 0; }
    bool daemonReady() const { return daemonConfigured_; }
    QString deviceInfo() const { return deviceInfo_; }

    bool hasCheckpoint() const { return !checkpointPath_.isEmpty(); }
    QString checkpointPath() const { return checkpointPath_; }
    void setCheckpointPath(const QString& path);
    QString checkpointDir() const { return checkpointDir_; }
    void setCheckpointDir(const QString& dir);
    QString activeCheckpointName() const;
    QVariantList availableCheckpoints() const { return availableCheckpoints_; }

    QString creativityPreset() const { return creativityPreset_; }
    void setCreativityPreset(const QString& preset);
    QVariantList creativityPresetOptions() const;

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

    int topK() const { return topK_; }
    void setTopK(int value);

    double topP() const { return topP_; }
    void setTopP(double value);

    double repetitionPenalty() const { return repetitionPenalty_; }
    void setRepetitionPenalty(double value);

    int maxSlideJudgeAreas() const { return maxSlideJudgeAreas_; }
    void setMaxSlideJudgeAreas(int value);

    int seed() const { return seed_; }
    void setSeed(int value);

    int debounceMs() const { return debounceMs_; }
    void setDebounceMs(int value);

    QString pythonMode() const { return pythonMode_; }
    void setPythonMode(const QString& mode);
    QVariantList envModeOptions() const;
    QString customPythonPath() const { return customPythonPath_; }
    void setCustomPythonPath(const QString& path);
    QString resolvedPythonPath() const;

    QString sourceMode() const { return sourceMode_; }
    void setSourceMode(const QString& mode);
    QVariantList sourceModeOptions() const;
    QString customSourceRoot() const { return customSourceRoot_; }
    void setCustomSourceRoot(const QString& path);
    QString resolvedSourceRoot() const;

    QString audioCacheSizeText() const { return audioCacheSizeText_; }

    bool ghostActive() const { return ghostActive_; }
    QString ghostText() const { return ghostText_; }
    QString ghostFirstLine() const { return ghostFirstLine_; }
    QString ghostExtraLines() const { return ghostExtraLines_; }
    int ghostReplaceStart() const { return ghostReplaceStart_; }
    int ghostReplaceEnd() const { return ghostReplaceEnd_; }
    int ghostStepCount() const { return static_cast<int>(ghostSteps_.size()); }
    int ghostCurrentStep() const { return ghostCurrentStep_; }

    bool inlineCardVisible() const { return inlineCardVisible_; }
    void setInlineCardVisible(bool visible);
    int inlineSpanBeats() const { return inlineSpanBeats_; }
    void setInlineSpanBeats(int beats);
    bool inlineFillToNext() const { return inlineFillToNext_; }
    void setInlineFillToNext(bool fill);
    QString inlineLevelOverride() const { return inlineLevelOverride_; }
    void setInlineLevelOverride(const QString& level);
    QString inlineDesignerOverride() const { return inlineDesignerOverride_; }
    void setInlineDesignerOverride(const QString& designer);

    bool pendingRangeActive() const { return pendingRangeActive_; }
    QString pendingRangeText() const { return pendingRangeText_; }
    int pendingReplaceStart() const { return pendingReplaceStart_; }
    int pendingReplaceEnd() const { return pendingReplaceEnd_; }
    double pendingStartSecond() const { return pendingStartSecond_; }
    double pendingEndSecond() const { return pendingEndSecond_; }
    int pendingNoteCount() const { return pendingNoteCount_; }

    bool emptyCardVisible() const;
    bool fullStreamingActive() const { return fullStreamingActive_; }
    double streamingProgress() const { return streamingProgress_; }
    int streamingBeat() const { return streamingBeat_; }
    int streamingTotalBeats() const { return streamingTotalBeats_; }

    // ---- Editor & UI Invokables ----
    Q_INVOKABLE void bindEditor(QObject* editorObject);
    Q_INVOKABLE void notifyEditorCursorOrTextChanged(int cursorPos, int selStart, int selEnd, bool textChanged);
    Q_INVOKABLE void notifyEditorCursorOrTextChanged(
        int cursorPos,
        int selStart,
        int selEnd,
        const QString& text,
        bool imeComposing = false,
        bool completionActive = false,
        bool followPlaybackActive = false);
    Q_INVOKABLE bool handleEditorKeyPress(int key, int modifiers);

    Q_INVOKABLE void triggerManualContinue();
    Q_INVOKABLE bool acceptFullGhost();
    Q_INVOKABLE bool acceptGhostAll() { return acceptFullGhost(); }
    Q_INVOKABLE bool acceptNextGhostStep();
    Q_INVOKABLE bool acceptGhostNextStep() { return acceptNextGhostStep(); }
    Q_INVOKABLE void dismissGhost();

    Q_INVOKABLE void openInlineCard();
    Q_INVOKABLE void closeInlineCard();
    Q_INVOKABLE void requestRangeGeneration(
        int spanBeats = -1,
        bool fillToNext = false,
        const QString& levelOverride = QString(),
        const QString& designerOverride = QString());
    Q_INVOKABLE void triggerRangeGenerate();
    Q_INVOKABLE bool acceptPendingRange();
    Q_INVOKABLE void regeneratePendingRange();
    Q_INVOKABLE void auditionPendingRange();
    Q_INVOKABLE void rejectPendingRange();

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
    Q_INVOKABLE void browseCustomSourceRoot();
    Q_INVOKABLE void clearAudioCache();
    Q_INVOKABLE void releaseVram();
    Q_INVOKABLE void unloadModelMemory() { releaseVram(); }
    Q_INVOKABLE void restartDaemon();
    Q_INVOKABLE void refreshAudioCacheSize();

signals:
    void configChanged();
    void statusChanged();
    void checkpointsChanged();
    void designerOptionsChanged();
    void cacheInfoChanged();
    void ghostStateChanged();
    void inlineCardChanged();
    void pendingRangeChanged();
    void emptyCardChanged();
    void streamingProgressChanged();

private:
    void loadPreferences();
    void savePreferences() const;
    void applyCreativityPresetValues(const QString& preset);
    void rebuildDesignerOptions(const QStringList& vocab);
    void updateStatus(const QString& state, const QString& label, const QString& detail = QString());

    QString defaultBuiltinRoot() const;
    QString defaultCheckpointsDir() const;
    QString resolvedResourcesDir() const;
    QString resolvedSlideDataPath() const;
    QString resolvedGeometryDir() const;
    QString resolvedStatsDir() const;
    QString resolvedCacheDir() const;
    QString resolvedFfmpegPath() const;

    void ensureDaemonRunning();
    void stopDaemon();
    void sendConfigureRequest();
    void sendBindSongIfNeeded();
    void sendSyncChartIfNeeded();
    void sendSuggestRequest(
        const QString& mode,
        int spanBeats,
        bool fillToNext,
        const QString& levelOverride,
        const QString& designerOverride);
    void writeDaemonJson(const QJsonObject& payload);
    void handleDaemonReadyReadStdout();
    void handleDaemonReadyReadStderr();
    void handleDaemonMessage(const QJsonObject& message);

    void updateGhostSplitLines();
    void applyDualSurfaceGhostPreview(
        const QString& ghostSimai,
        double startSecond,
        double activeBpm);
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

    // ---- Preferences ----
    bool enabled_ = true;
    bool autoContinue_ = true;
    QString creativityPreset_ = QStringLiteral("balanced");
    QString defaultDesigner_;
    QString device_ = QStringLiteral("auto");
    QString precision_ = QStringLiteral("bf16");
    double temperature_ = 1.0;
    int topK_ = 32;
    double topP_ = 0.95;
    double repetitionPenalty_ = 1.0;
    int maxSlideJudgeAreas_ = 4;
    int seed_ = -1;
    int debounceMs_ = 420;
    QString checkpointPath_;
    QString checkpointDir_;
    QString pythonMode_ = QStringLiteral("bundled");
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
    QString activeRequestMode_;
    quint64 activeRequestRevision_ = 0;
    QString boundSongKey_;
    quint64 syncedChartRevision_ = 0;
    QString lastObservedEditorText_;

    QTimer debounceTimer_;
    QTimer idleVramTimer_;
    int lastCursorPos_ = 0;
    int lastSelStart_ = 0;
    int lastSelEnd_ = 0;

    // ---- Ghost State ----
    bool ghostActive_ = false;
    QString ghostText_;
    QString ghostFirstLine_;
    QString ghostExtraLines_;
    int ghostReplaceStart_ = 0;
    int ghostReplaceEnd_ = 0;
    QJsonArray ghostSteps_;
    int ghostCurrentStep_ = 0;

    // ---- Inline Card & Pending Range State ----
    bool inlineCardVisible_ = false;
    int inlineSpanBeats_ = 4;
    bool inlineFillToNext_ = false;
    QString inlineLevelOverride_;
    QString inlineDesignerOverride_;
    bool pendingRangeActive_ = false;
    QString pendingRangeText_;
    int pendingReplaceStart_ = 0;
    int pendingReplaceEnd_ = 0;
    double pendingStartSecond_ = 0.0;
    double pendingEndSecond_ = 0.0;
    int pendingNoteCount_ = 0;
    int lastRangeSpanBeats_ = 4;
    bool lastRangeFillToNext_ = false;
    QString lastRangeLevelOverride_;
    QString lastRangeDesignerOverride_;

    // ---- Empty Card & Full Streaming State ----
    int dismissedEmptyDifficultyId_ = -1;
    bool fullStreamingActive_ = false;
    double streamingProgress_ = 0.0;
    int streamingBeat_ = 0;
    int streamingTotalBeats_ = 0;
    QString fullStreamingInitialText_;
};

} // namespace miacode::ui
