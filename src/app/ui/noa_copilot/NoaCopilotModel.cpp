#include "app/ui/noa_copilot/NoaCopilotModel.h"

#include "app/services/PreferenceDocument.h"
#include "app/services/UiRequestService.h"
#include "app/ui/document/DocumentModel.h"
#include "app/ui/preview/PreviewModel.h"
#include "app/ui/timeline/TimelineModel.h"
#include "common/DebugLog.h"
#include "core/chart/ChartAssetPaths.h"
#include "core/chart/parser/SimaiParser.h"
#include "core/chart/model/TimelineMarkerOffset.h"
#include "preview/runtime/PreviewRuntime.h"
#include "timeline/TimelineQuickModel.h"
#include "timeline/quick/TimelineQuickStateBridge.h"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QMetaObject>
#include <QScopedValueRollback>
#include <QThread>
#include <memory>
#include "app/ui/noa_copilot/NoaEnvironments.h"
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QUrl>
#include <QtMath>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace miacode::ui {

namespace {

constexpr auto kPrefSectionKey = "noa_copilot";
constexpr int kIdleVramReleaseMs = 5 * 60 * 1000; // 5 minutes

QString formatBytesText(qint64 bytes)
{
    if (bytes <= 0) {
        return QStringLiteral("0 MB");
    }
    const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    if (mb < 1024.0) {
        return QStringLiteral("%1 MB").arg(mb, 0, 'f', 1);
    }
    return QStringLiteral("%1 GB").arg(mb / 1024.0, 0, 'f', 2);
}

QString normalizeModeToken(const QString& rawMode)
{
    const QString trimmed = rawMode.trimmed().toLower();
    if (trimmed == QStringLiteral("custom")) {
        return QStringLiteral("custom");
    }
    return QStringLiteral("bundled");
}

std::pair<int, int> utf16OffsetToLineCol(const QString& text, int offset)
{
    const int bounded = qBound(0, offset, int(text.size()));
    int line = 1;
    int col = 1;
    for (int i = 0; i < bounded; ++i) {
        if (text.at(i) == QLatin1Char('\n')) {
            ++line;
            col = 1;
        } else {
            ++col;
        }
    }
    return {line, col};
}

void compileGhost(const QString& text, double start, double bpm, QVector<TimelineNoteMarker>& markers,
                  TimelineRenderSnapshot& timeline)
{
    QString chart = text.trimmed();
    if (chart.isEmpty()) { markers = {}; timeline = {}; return; }
    if (!chart.startsWith('(')) chart.prepend(QStringLiteral("(%1)").arg(bpm > 0 ? bpm : 160.0));
    if (!chart.endsWith('E', Qt::CaseInsensitive)) chart.append(QStringLiteral("\nE"));
    const auto parsed = SimaiParser::parseForTimeline(chart);
    using namespace miacode::timeline::offset;
    markers = shiftedNoteMarkers(parsed.noteMarkers, start, NonFiniteHandling::PassThrough);
    TimelineQuickModel model;
    model.rebuildFromText(chart, start, {});
    timeline = model.snapshot();
}
} // namespace

NoaCopilotModel::NoaCopilotModel(
    DocumentModel& document,
    PreviewModel& preview,
    TimelineModel& timeline,
    miacode::UiRequestService& uiRequests,
    QObject* parent)
    : QObject(parent)
    , document_(document)
    , preview_(preview)
    , timeline_(timeline)
    , uiRequests_(uiRequests)
{
    streamFlushTimer_.setSingleShot(true);
    streamFlushTimer_.setTimerType(Qt::PreciseTimer);
    connect(&streamFlushTimer_, &QTimer::timeout, this, [this] {
        const QString text = pendingStreamChart_;
        pendingStreamChart_.clear();
        if (fullStreamingActive_ && !text.isEmpty()) publishFullChart(text);
    });
    renderTimer_.setSingleShot(true);
    renderTimer_.setInterval(65);
    connect(&renderTimer_, &QTimer::timeout, this, &NoaCopilotModel::showCandidate);
    warmTimer_.setSingleShot(true);
    warmTimer_.setInterval(550);
    connect(&warmTimer_, &QTimer::timeout, this, &NoaCopilotModel::warmSong);
    connect(&document_, &DocumentModel::currentDifficultyFieldsChanged, this, [this] { warmTimer_.start(); });
    debounceTimer_.setSingleShot(true);
    connect(&debounceTimer_, &QTimer::timeout, this, [this]() {
        if (!enabled_ || !autoContinue_ || !hasCheckpoint() || busy() || inlineCardVisible_ || pendingRangeActive_) {
            return;
        }
        if (lastSelStart_ != lastSelEnd_) {
            return;
        }
        if (!document_.hasDocument() || document_.currentDifficultyId() <= 0 || document_.wholeSourceEditorActive()) {
            return;
        }
        sendSuggestRequest(QStringLiteral("continue"), 4, false, QString(), QString());
    });

    idleVramTimer_.setSingleShot(true);
    idleVramTimer_.setInterval(kIdleVramReleaseMs);
    connect(&idleVramTimer_, &QTimer::timeout, this, [this]() {
        if (device_ == QStringLiteral("cuda") && !activeRequestId_ && candidates_.isEmpty()) releaseVram();
    });

    // Ensure default checkpoints directory exists so users can drop .ckpt files into it.
    QDir().mkpath(defaultCheckpointsDir());
    QDir().mkpath(resolvedCacheDir());

    rebuildDesignerOptions({});
    loadPreferences();
    refreshCheckpoints();
    refreshAudioCacheSize();
    scanEnvironments(false);

    connect(&document_, &DocumentModel::documentStateChanged, this, [this]() {
        if (!document_.hasDocument() || document_.wholeSourceEditorActive()) {
            dismissGhost();
            if (pendingRangeActive_) {
                rejectPendingRange();
            }
        }
        sendBindSongIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::documentReplaced, this, [this]() {
        cancelActiveGeneration();
        discardCandidates();
        dismissGhost();
        if (pendingRangeActive_) {
            rejectPendingRange();
        }
        dismissedEmptyDifficultyId_ = -1;
        lastObservedEditorText_ = document_.chartText();
        sendBindSongIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::currentFilePathChanged, this, [this]() {
        sendBindSongIfNeeded();
    });
    connect(&document_, &DocumentModel::metadataChanged, this, [this]() {
        warmTimer_.start();
        sendBindSongIfNeeded();
    });
    connect(&document_, &DocumentModel::currentDifficultyChanged, this, [this]() {
        cancelActiveGeneration();
        discardCandidates();
        warmTimer_.start();
        dismissGhost();
        if (pendingRangeActive_) {
            rejectPendingRange();
        }
        dismissedEmptyDifficultyId_ = -1;
        lastObservedEditorText_ = document_.chartText();
        sendSyncChartIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::chartTextChanged, this, [this]() {
        if (!fullStreamingActive_) warmTimer_.start();
        emit emptyCardChanged();
    });

    if (enabled_) {
        QTimer::singleShot(300, this, [this]() {
            if (hasCheckpoint() || document_.hasDocument()) ensureDaemonRunning();
        });
    } else {
        updateStatus(QStringLiteral("disabled"), QStringLiteral("已关闭"));
    }
}

NoaCopilotModel::~NoaCopilotModel()
{
    stopDaemon();
#ifdef Q_OS_WIN
    if (windowsJobHandle_ != nullptr) {
        CloseHandle(static_cast<HANDLE>(windowsJobHandle_));
        windowsJobHandle_ = nullptr;
    }
#endif
}

QString NoaCopilotModel::defaultBuiltinRoot() const
{
    return QDir::cleanPath(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("noa-copilot")));
}

QString NoaCopilotModel::defaultCheckpointsDir() const
{
    return QDir::cleanPath(QDir(defaultBuiltinRoot()).filePath(QStringLiteral("checkpoints")));
}

QString NoaCopilotModel::resolvedPythonPath() const
{
    if (device_ == QStringLiteral("cuda")) return cudaPython_;
    if (!cpuPython_.isEmpty()) return cpuPython_;
    return QDir(defaultBuiltinRoot()).filePath(QStringLiteral("env/python.exe"));
}

QString NoaCopilotModel::resolvedSourceRoot() const
{
    if (sourceMode_ == QStringLiteral("custom") && !customSourceRoot_.trimmed().isEmpty()) {
        const QString custom = QDir::cleanPath(customSourceRoot_.trimmed());
        const QString srcSubdir = QDir(custom).filePath(QStringLiteral("src"));
        if (QFileInfo::exists(QDir(srcSubdir).filePath(QStringLiteral("maidata")))) {
            return QDir::cleanPath(srcSubdir);
        }
        return custom;
    }
    return QDir(defaultBuiltinRoot()).filePath(QStringLiteral("runtime-src"));
}

QString NoaCopilotModel::resolvedCacheDir() const
{
    const QString appData = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    if (!appData.isEmpty()) {
        return QDir::cleanPath(QDir(appData).filePath(QStringLiteral("noa-copilot/cache")));
    }
    return QDir::cleanPath(QDir(defaultBuiltinRoot()).filePath(QStringLiteral("cache")));
}

QVariantList NoaCopilotModel::deviceModeOptions() const
{
    return {QVariantMap{{"value","cpu"},{"label","CPU"},{"enabled",cpuAvailable_}},
            QVariantMap{{"value","cuda"},{"label","CUDA"},{"enabled",cudaAvailable_}}};
}

QVariantList NoaCopilotModel::precisionModeOptions() const
{
    return {
        QVariantMap{{QStringLiteral("value"), QStringLiteral("bf16")}, {QStringLiteral("label"), QStringLiteral("BF16 (推荐)")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("fp16")}, {QStringLiteral("label"), QStringLiteral("FP16")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("fp32")}, {QStringLiteral("label"), QStringLiteral("FP32")}},
    };
}

QVariantList NoaCopilotModel::envModeOptions() const
{
    return {
        QVariantMap{{QStringLiteral("value"), QStringLiteral("bundled")}, {QStringLiteral("label"), QStringLiteral("MiaCode 内置 PyTorch 环境 (推荐)")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("custom")}, {QStringLiteral("label"), QStringLiteral("自定义 Python 解释器")}},
    };
}

QVariantList NoaCopilotModel::sourceModeOptions() const
{
    QString bundledLabel = QStringLiteral("MiaCode 内置 Flat320 推理源码");
    QFile runtimeManifest(QDir(defaultBuiltinRoot()).filePath(QStringLiteral("runtime.json")));
    if (runtimeManifest.open(QIODevice::ReadOnly)) {
        const QString architecture = QJsonDocument::fromJson(runtimeManifest.readAll())
                                         .object().value(QStringLiteral("architecture")).toString();
        const qsizetype version = architecture.lastIndexOf(QStringLiteral(".v"));
        if (version >= 0) {
            bundledLabel += QStringLiteral(" (%1)").arg(architecture.mid(version + 1));
        }
    }
    return {
        QVariantMap{{QStringLiteral("value"), QStringLiteral("bundled")}, {QStringLiteral("label"), bundledLabel}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("custom")}, {QStringLiteral("label"), QStringLiteral("自定义 Worktree / 推理源码目录")}},
    };
}

void NoaCopilotModel::loadPreferences()
{
    const QJsonObject root = PreferenceDocument::loadPreferencesObject();
    const QJsonObject sec = root.value(QLatin1String(kPrefSectionKey)).toObject();
    lineAlignBeats_ = sec["line_align_beats"].toInt(4);
    candidateCount_ = sec["candidate_count"].toInt(3);
    cpuThreads_ = sec["cpu_threads"].toInt(4);
    mertCache_ = sec["mert_cache"].toBool(true);
    roformerCache_ = sec["roformer_cache"].toBool(true);
    mertDevice_ = sec["mert_device"].toString(QStringLiteral("cpu"));
    cpuPython_ = sec["cpu_python"].toString();
    cudaPython_ = sec["cuda_python"].toString();
    enabled_ = sec.value(QStringLiteral("enabled")).toBool(true);
    autoContinue_ = sec.value(QStringLiteral("auto_continue")).toBool(true);
    defaultDesigner_ = sec.value(QStringLiteral("default_designer")).toString();
    device_ = sec.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    precision_ = sec.value(QStringLiteral("precision")).toString(QStringLiteral("bf16"));
    temperature_ = qBound(0.0, sec.value(QStringLiteral("temperature")).toDouble(0.5), 1.0);
    slideBias_ = sec.value(QStringLiteral("slide_bias")).toDouble(0.0);
    seed_ = sec.value(QStringLiteral("seed")).toInt(-1);
    debounceMs_ = qBound(150, sec.value(QStringLiteral("debounce_ms")).toInt(420), 3000);
    checkpointPath_ = sec.value(QStringLiteral("checkpoint_path")).toString();
    checkpointDir_ = sec.value(QStringLiteral("checkpoint_dir")).toString();
    pythonMode_ = normalizeModeToken(sec.value(QStringLiteral("python_mode")).toString(QStringLiteral("bundled")));
    QFile runtimeSettings(QDir(defaultBuiltinRoot()).filePath(QStringLiteral("wsl.json")));
    QJsonObject defaults;
    if (runtimeSettings.open(QIODevice::ReadOnly))
        defaults = QJsonDocument::fromJson(runtimeSettings.readAll()).object();
    const QString bundledPython = defaults.value(QStringLiteral("python")).toString();
    wslPython_ = sec.value(QStringLiteral("wsl_python")).toString(bundledPython);
    wslDistribution_ = sec.value(QStringLiteral("wsl_distribution")).toString(defaults.value(QStringLiteral("distribution")).toString());
    customPythonPath_ = sec.value(QStringLiteral("custom_python_path")).toString();
    sourceMode_ = normalizeModeToken(sec.value(QStringLiteral("source_mode")).toString(QStringLiteral("bundled")));
    customSourceRoot_ = sec.value(QStringLiteral("custom_source_root")).toString();
    if (device_ == "auto") device_ = "cpu";
    if (cpuPython_.isEmpty() && pythonMode_ == "custom") cpuPython_ = customPythonPath_;
    if (!sec.contains("cuda_python")) cudaPython_ = device_ == "cuda" && pythonMode_ == "custom" ? customPythonPath_ : wslPython_;
    // The previous Windows bundle saved its default interpreter in user preferences.
    if (sourceMode_ == QStringLiteral("bundled") && pythonMode_ == QStringLiteral("bundled")
        && !bundledPython.isEmpty() && wslDistribution_ == defaults.value(QStringLiteral("distribution")).toString()
        && (wslPython_.isEmpty() || wslPython_ == defaults.value(QStringLiteral("previous_python")).toString())) {
        wslPython_ = bundledPython;
    }
}

void NoaCopilotModel::savePreferences() const
{
    QJsonObject root = PreferenceDocument::loadPreferencesObject();
    QJsonObject sec;
    sec["line_align_beats"] = lineAlignBeats_;
    sec["candidate_count"] = candidateCount_;
    sec["cpu_threads"] = cpuThreads_;
    sec["mert_cache"] = mertCache_;
    sec["roformer_cache"] = roformerCache_;
    sec["mert_device"] = mertDevice_;
    sec["cpu_python"] = cpuPython_;
    sec["cuda_python"] = cudaPython_;
    sec.insert(QStringLiteral("enabled"), enabled_);
    sec.insert(QStringLiteral("auto_continue"), autoContinue_);
    sec.insert(QStringLiteral("default_designer"), defaultDesigner_);
    sec.insert(QStringLiteral("device"), device_);
    sec.insert(QStringLiteral("precision"), precision_);
    sec.insert(QStringLiteral("temperature"), temperature_);
    sec.insert(QStringLiteral("slide_bias"), slideBias_);
    sec.insert(QStringLiteral("seed"), seed_);
    sec.insert(QStringLiteral("debounce_ms"), debounceMs_);
    sec.insert(QStringLiteral("checkpoint_path"), checkpointPath_);
    sec.insert(QStringLiteral("checkpoint_dir"), checkpointDir_);
    sec.insert(QStringLiteral("python_mode"), pythonMode_);
    sec.insert(QStringLiteral("wsl_python"), wslPython_);
    sec.insert(QStringLiteral("wsl_distribution"), wslDistribution_);
    sec.insert(QStringLiteral("custom_python_path"), customPythonPath_);
    sec.insert(QStringLiteral("source_mode"), sourceMode_);
    sec.insert(QStringLiteral("custom_source_root"), customSourceRoot_);
    root.insert(QLatin1String(kPrefSectionKey), sec);
    PreferenceDocument::savePreferencesObject(root);
}

void NoaCopilotModel::rebuildDesignerOptions(const QStringList& vocab)
{
    QVariantList list;
    list.append(QVariantMap{
        {QStringLiteral("value"), QString()},
        {QStringLiteral("label"), QStringLiteral("跟随当前谱师 / 自动")},
    });
    for (const QString& entry : vocab) {
        if (entry.trimmed().isEmpty()) {
            continue;
        }
        list.append(QVariantMap{
            {QStringLiteral("value"), entry},
            {QStringLiteral("label"), entry},
        });
    }
    designerOptions_ = list;
    emit designerOptionsChanged();
}

void NoaCopilotModel::updateStatus(const QString& state, const QString& label, const QString& detail)
{
    if (statusState_ == state && statusLabel_ == label && statusDetail_ == detail) {
        return;
    }
    statusState_ = state;
    statusLabel_ = label;
    statusDetail_ = detail;
    emit statusChanged();
}

void NoaCopilotModel::setEnabled(bool value)
{
    if (enabled_ == value) {
        return;
    }
    enabled_ = value;
    savePreferences();
    emit configChanged();
    emit emptyCardChanged();
    if (!enabled_) {
        dismissGhost();
        if (pendingRangeActive_) {
            rejectPendingRange();
        }
        stopDaemon();
        updateStatus(QStringLiteral("disabled"), QStringLiteral("已关闭"));
    } else {
        ensureDaemonRunning();
    }
}

void NoaCopilotModel::setAutoContinue(bool value)
{
    if (autoContinue_ == value) {
        return;
    }
    autoContinue_ = value;
    savePreferences();
    if (!autoContinue_) {
        debounceTimer_.stop();
    }
    emit configChanged();
}

void NoaCopilotModel::setCheckpointPath(const QString& path)
{
    const QString cleaned = path.trimmed().isEmpty() ? QString() : QDir::cleanPath(path.trimmed());
    if (checkpointPath_ == cleaned) {
        return;
    }
    checkpointPath_ = cleaned;
    savePreferences();
    refreshCheckpoints();
    if (enabled_) {
        ensureDaemonRunning();
        sendConfigureRequest();
    }
}

void NoaCopilotModel::setCheckpointDir(const QString& dir)
{
    const QString cleaned = dir.trimmed().isEmpty() ? QString() : QDir::cleanPath(dir.trimmed());
    if (checkpointDir_ == cleaned) {
        return;
    }
    checkpointDir_ = cleaned;
    savePreferences();
    refreshCheckpoints();
}

QString NoaCopilotModel::activeCheckpointName() const
{
    if (checkpointPath_.isEmpty()) {
        return QStringLiteral("未选择模型权重");
    }
    return QFileInfo(checkpointPath_).fileName();
}

void NoaCopilotModel::setDefaultDesigner(const QString& designer)
{
    if (defaultDesigner_ == designer) {
        return;
    }
    defaultDesigner_ = designer;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setDevice(const QString& value)
{
    if (device_ == value) {
        return;
    }
    discardCandidates();
    device_ = value;
    savePreferences();
    emit configChanged();
    if (enabled_) {
        restartDaemon();
    }
}

void NoaCopilotModel::setPrecision(const QString& value)
{
    if (precision_ == value) {
        return;
    }
    precision_ = value;
    savePreferences();
    emit configChanged();
    if (enabled_) {
        sendConfigureRequest();
    }
}

void NoaCopilotModel::setTemperature(double value)
{
    value = qBound(0.0, value, 1.0);
    if (qFuzzyCompare(temperature_, value)) {
        return;
    }
    temperature_ = value;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setLineAlignBeats(int value)
{
    value = qBound(1, value, 64);
    if (lineAlignBeats_ == value) return;
    lineAlignBeats_ = value;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setCandidateCount(int value)
{
    value = qBound(1, value, 8);
    if (candidateCount_ == value) return;
    candidateCount_ = value;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setCpuThreads(int value)
{
    value = qBound(1, value, 64);
    if (cpuThreads_ == value) return;
    cpuThreads_ = value;
    savePreferences();
    emit configChanged();
    if (enabled_) restartDaemon();
}

void NoaCopilotModel::setMertCache(bool value)
{
    if (mertCache_ == value) return;
    mertCache_ = value;
    savePreferences();
    emit configChanged();
    warmTimer_.start();
}

void NoaCopilotModel::setRoformerCache(bool value)
{
    if (roformerCache_ == value) return;
    roformerCache_ = value;
    savePreferences();
    emit configChanged();
    warmTimer_.start();
}

void NoaCopilotModel::setMertDevice(QString value)
{
    if (mertDevice_ == value) return;
    mertDevice_ = value;
    savePreferences();
    emit configChanged();
    if (enabled_) restartDaemon();
}

void NoaCopilotModel::setCpuPython(QString value)
{
    if (cpuPython_ == value) return;
    cpuPython_ = value;
    savePreferences();
    emit configChanged();
    scanEnvironments(false);
    if (enabled_) restartDaemon();
}

void NoaCopilotModel::setCudaPython(QString value)
{
    if (cudaPython_ == value) return;
    cudaPython_ = value;
    savePreferences();
    emit configChanged();
    scanEnvironments(false);
    if (enabled_) restartDaemon();
}

void NoaCopilotModel::setSlideBias(double value)
{
    if (qFuzzyCompare(slideBias_, value)) return;
    slideBias_ = value;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setSeed(int value)
{
    if (seed_ == value) {
        return;
    }
    seed_ = value;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setDebounceMs(int value)
{
    value = qBound(150, value, 3000);
    if (debounceMs_ == value) {
        return;
    }
    debounceMs_ = value;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setPythonMode(const QString& mode)
{
    const QString normalized = normalizeModeToken(mode);
    if (pythonMode_ == normalized) {
        return;
    }
    pythonMode_ = normalized;
    savePreferences();
    emit configChanged();
    if (enabled_) {
        restartDaemon();
    }
}

bool NoaCopilotModel::usesWsl() const
{
#ifdef Q_OS_WIN
    return resolvedPythonPath().startsWith('/');
#else
    return false;
#endif
}

void NoaCopilotModel::setWslPython(const QString& path)
{
    if (wslPython_ == path.trimmed()) return;
    wslPython_ = path.trimmed();
    savePreferences();
    emit configChanged();
    if (enabled_ && usesWsl()) restartDaemon();
}

void NoaCopilotModel::setWslDistribution(const QString& name)
{
    if (wslDistribution_ == name.trimmed()) return;
    wslDistribution_ = name.trimmed();
    savePreferences();
    emit configChanged();
    scanEnvironments(false);
    if (enabled_) restartDaemon();
}

void NoaCopilotModel::setCustomPythonPath(const QString& path)
{
    if (customPythonPath_ == path) {
        return;
    }
    customPythonPath_ = path;
    savePreferences();
    emit configChanged();
    if (enabled_ && pythonMode_ == QStringLiteral("custom")) {
        restartDaemon();
    }
}

void NoaCopilotModel::setSourceMode(const QString& mode)
{
    const QString normalized = normalizeModeToken(mode);
    if (sourceMode_ == normalized) {
        return;
    }
    sourceMode_ = normalized;
    savePreferences();
    emit configChanged();
    if (enabled_) {
        restartDaemon();
    }
}

void NoaCopilotModel::setCustomSourceRoot(const QString& path)
{
    if (customSourceRoot_ == path) {
        return;
    }
    customSourceRoot_ = path;
    savePreferences();
    emit configChanged();
    if (enabled_ && sourceMode_ == QStringLiteral("custom")) {
        restartDaemon();
    }
}

void NoaCopilotModel::setInlineCardVisible(bool visible)
{
    if (inlineCardVisible_ == visible) {
        return;
    }
    inlineCardVisible_ = visible;
    emit inlineCardChanged();
}

void NoaCopilotModel::setInlineSpanBeats(int beats)
{
    beats = qBound(0, beats, 64);
    const bool fill = (beats == 0);
    if (inlineSpanBeats_ == beats && inlineFillToNext_ == fill) {
        return;
    }
    inlineSpanBeats_ = beats;
    inlineFillToNext_ = fill;
    emit inlineCardChanged();
}

void NoaCopilotModel::setInlineFillToNext(bool fill)
{
    if (inlineFillToNext_ == fill) {
        return;
    }
    inlineFillToNext_ = fill;
    emit inlineCardChanged();
}

void NoaCopilotModel::setInlineLevelOverride(const QString& level)
{
    if (inlineLevelOverride_ == level) {
        return;
    }
    inlineLevelOverride_ = level;
    emit inlineCardChanged();
}

void NoaCopilotModel::setInlineDesignerOverride(const QString& designer)
{
    if (inlineDesignerOverride_ == designer) {
        return;
    }
    inlineDesignerOverride_ = designer;
    emit inlineCardChanged();
}

bool NoaCopilotModel::emptyCardVisible() const
{
    if (!enabled_ || !document_.hasDocument() || document_.wholeSourceEditorActive()) {
        return false;
    }
    const int diffId = document_.currentDifficultyId();
    if (diffId <= 0 || diffId == dismissedEmptyDifficultyId_) {
        return false;
    }
    if (fullStreamingActive_ || pendingRangeActive_ || ghostActive_) {
        return false;
    }
    QString trimmed = document_.chartText().trimmed();
    if (trimmed.compare(QStringLiteral("E"), Qt::CaseInsensitive) == 0) {
        trimmed.clear();
    }
    return trimmed.isEmpty();
}

void NoaCopilotModel::refreshCheckpoints()
{
    QStringList dirsToScan;
    dirsToScan.append(defaultCheckpointsDir());
    if (!checkpointDir_.isEmpty() && !dirsToScan.contains(checkpointDir_)) {
        dirsToScan.append(checkpointDir_);
    }

    QVariantList discovered;
    QStringList seenPaths;
    auto addFile = [&](const QString& filePath) {
        const QFileInfo info(filePath);
        if (!info.exists() || !info.isFile()) {
            return;
        }
        const QString canonical = QDir::cleanPath(info.absoluteFilePath());
        if (seenPaths.contains(canonical, Qt::CaseInsensitive)) {
            return;
        }
        seenPaths.append(canonical);
        discovered.append(QVariantMap{
            {QStringLiteral("path"), canonical},
            {QStringLiteral("value"), canonical},
            {QStringLiteral("name"), info.fileName()},
            {QStringLiteral("label"), info.fileName()},
        });
    };

    if (!checkpointPath_.isEmpty()) {
        addFile(checkpointPath_);
    }
    for (const QString& dirPath : dirsToScan) {
        QDir dir(dirPath);
        if (!dir.exists()) {
            continue;
        }
        const QFileInfoList files = dir.entryInfoList(
            {QStringLiteral("*.ckpt"), QStringLiteral("*.pt"), QStringLiteral("*.safetensors")},
            QDir::Files | QDir::Readable,
            QDir::Time);
        for (const QFileInfo& info : files) {
            addFile(info.absoluteFilePath());
        }
    }

    if (checkpointPath_.isEmpty() && !discovered.isEmpty()) {
        checkpointPath_ = discovered.constFirst().toMap().value(QStringLiteral("value")).toString();
        savePreferences();
    } else if (!checkpointPath_.isEmpty() && !QFileInfo::exists(checkpointPath_)) {
        checkpointPath_.clear();
        if (!discovered.isEmpty()) {
            checkpointPath_ = discovered.constFirst().toMap().value(QStringLiteral("value")).toString();
        }
        savePreferences();
    }

    availableCheckpoints_ = discovered;
    emit checkpointsChanged();
}

void NoaCopilotModel::openCheckpointFolder()
{
    const QString targetDir = !checkpointDir_.isEmpty() ? checkpointDir_ : defaultCheckpointsDir();
    QDir().mkpath(targetDir);
    QDesktopServices::openUrl(QUrl::fromLocalFile(targetDir));
}

void NoaCopilotModel::chooseCheckpointFileUrl(const QUrl& url)
{
    const QString local = url.isLocalFile() ? url.toLocalFile() : url.toString();
    if (!local.trimmed().isEmpty()) {
        setCheckpointPath(local);
    }
}

void NoaCopilotModel::chooseCustomPythonUrl(const QUrl& url)
{
    const QString local = url.isLocalFile() ? url.toLocalFile() : url.toString();
    if (!local.trimmed().isEmpty()) {
        setCustomPythonPath(local);
    }
}

void NoaCopilotModel::chooseCustomSourceDirUrl(const QUrl& url)
{
    const QString local = url.isLocalFile() ? url.toLocalFile() : url.toString();
    if (!local.trimmed().isEmpty()) {
        setCustomSourceRoot(local);
    }
}

void NoaCopilotModel::browseCheckpointFile()
{
    miacode::FileRequest req;
    req.title = QStringLiteral("选择 NoaCopilot 模型权重 (.ckpt)");
    req.startPath = !checkpointPath_.isEmpty() ? checkpointPath_ : defaultCheckpointsDir();
    req.nameFilters = {
        QStringLiteral("Model Checkpoints (*.ckpt *.pt)"),
        QStringLiteral("All Files (*.*)"),
    };
    uiRequests_.requestFile(req, [this](const QString& chosen) {
        if (!chosen.isEmpty()) {
            setCheckpointPath(chosen);
        }
    });
}

void NoaCopilotModel::browseCheckpointDirectory()
{
    miacode::FileRequest req;
    req.title = QStringLiteral("选择模型权重目录");
    req.startPath = !checkpointDir_.isEmpty() ? checkpointDir_ : defaultCheckpointsDir();
    req.selectFolder = true;
    uiRequests_.requestFile(req, [this](const QString& chosen) {
        if (!chosen.isEmpty()) {
            setCheckpointDir(chosen);
        }
    });
}

void NoaCopilotModel::browseCustomPythonPath()
{
    miacode::FileRequest req;
    req.title = QStringLiteral("选择自定义 Python 解释器 (python.exe)");
    req.startPath = !customPythonPath_.isEmpty() ? customPythonPath_ : resolvedPythonPath();
    req.nameFilters = {
        QStringLiteral("Python Executable (python.exe python)"),
        QStringLiteral("All Files (*.*)"),
    };
    uiRequests_.requestFile(req, [this](const QString& chosen) {
        if (!chosen.isEmpty()) {
            setCustomPythonPath(chosen);
        }
    });
}

void NoaCopilotModel::browseCustomSourceRoot()
{
    miacode::FileRequest req;
    req.title = QStringLiteral("选择自定义推理源码根目录 (worktree 或 src)");
    req.startPath = !customSourceRoot_.isEmpty() ? customSourceRoot_ : resolvedSourceRoot();
    req.selectFolder = true;
    uiRequests_.requestFile(req, [this](const QString& chosen) {
        if (!chosen.isEmpty()) {
            setCustomSourceRoot(chosen);
        }
    });
}

void NoaCopilotModel::refreshAudioCacheSize()
{
    qint64 totalBytes = 0;
    if (document_.currentFilePath().isEmpty()) { audioCacheSizeText_ = "0 MB"; emit cacheInfoChanged(); return; }
    const QDir project(QFileInfo(document_.currentFilePath()).dir().filePath(".miacode/.noacopilot"));
    QDirIterator it(project.absolutePath(), {"mert_*.pt","roformer_*.pt","drafter_*.pt"}, QDir::Files);
    while (it.hasNext()) {
        it.next();
        totalBytes += it.fileInfo().size();
    }
    const QString nextText = formatBytesText(totalBytes);
    if (audioCacheSizeText_ != nextText) {
        audioCacheSizeText_ = nextText;
        emit cacheInfoChanged();
    }
}

void NoaCopilotModel::clearAudioCache()
{
    if (daemonProcess_ != nullptr && daemonProcess_->state() == QProcess::Running) {
        QJsonObject cmd;
        cmd.insert(QStringLiteral("op"), QStringLiteral("clear_cache"));
        cmd.insert(QStringLiteral("id"), nextRpcId_++);
        writeDaemonJson(cmd);
    }
    if (!document_.currentFilePath().isEmpty()) {
        const QDir cacheDir(QFileInfo(document_.currentFilePath()).dir().filePath(".miacode/.noacopilot"));
        for (const auto& name : cacheDir.entryList({"mert_*.pt","roformer_*.pt","drafter_*.pt","manifest.json"},QDir::Files))
            QFile::remove(cacheDir.filePath(name));
    }
    boundSongKey_.clear();
    refreshAudioCacheSize();
}

void NoaCopilotModel::releaseVram()
{
    if (daemonProcess_ != nullptr && daemonProcess_->state() == QProcess::Running) {
        QJsonObject cmd;
        cmd.insert(QStringLiteral("op"), QStringLiteral("release_vram"));
        cmd.insert(QStringLiteral("id"), nextRpcId_++);
        writeDaemonJson(cmd);
    }
}

void NoaCopilotModel::restartDaemon()
{
    dismissGhost();
    stopDaemon();
    if (enabled_) {
        ensureDaemonRunning();
    }
}

void NoaCopilotModel::ensureDaemonRunning()
{
    if (!enabled_) {
        return;
    }
    if (daemonProcess_ != nullptr && daemonProcess_->state() != QProcess::NotRunning) {
        return;
    }

    const QString pythonExe = resolvedPythonPath();
    const QString sourceRoot = resolvedSourceRoot();
    if (!usesWsl() && !QFileInfo::exists(pythonExe)) {
        updateStatus(
            QStringLiteral("error"),
            QStringLiteral("未找到 Python 环境"),
            QStringLiteral("请检查内置环境或高级设置中的 Python 路径: %1").arg(pythonExe));
        return;
    }
    if (!QFileInfo::exists(QDir(sourceRoot).filePath(QStringLiteral("maidata")))) {
        updateStatus(
            QStringLiteral("error"),
            QStringLiteral("未找到推理源码"),
            QStringLiteral("请检查内置 runtime-src 或高级设置中的源码路径: %1").arg(sourceRoot));
        return;
    }

    if (daemonProcess_ != nullptr) {
        daemonProcess_->deleteLater();
        daemonProcess_ = nullptr;
    }

    daemonProcess_ = new QProcess(this);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
    env.insert(QStringLiteral("OMP_NUM_THREADS"), QString::number(cpuThreads_));
    env.insert(QStringLiteral("PYTHONUTF8"), QStringLiteral("1"));
    env.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    env.insert(QStringLiteral("PYTHONPATH"), QDir::toNativeSeparators(sourceRoot));
    daemonProcess_->setProcessEnvironment(env);
    daemonProcess_->setWorkingDirectory(sourceRoot);

    connect(daemonProcess_, &QProcess::readyReadStandardOutput, this, &NoaCopilotModel::handleDaemonReadyReadStdout);
    connect(daemonProcess_, &QProcess::readyReadStandardError, this, &NoaCopilotModel::handleDaemonReadyReadStderr);
    connect(daemonProcess_, &QProcess::started, this, [this]() {
#ifdef Q_OS_WIN
        if (windowsJobHandle_ == nullptr) {
            HANDLE job = CreateJobObjectW(nullptr, nullptr);
            if (job != nullptr) {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = {};
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info));
                windowsJobHandle_ = job;
            }
        }
        if (windowsJobHandle_ != nullptr && daemonProcess_ != nullptr) {
            const qint64 pid = daemonProcess_->processId();
            if (pid > 0) {
                HANDLE hProc = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(pid));
                if (hProc != nullptr) {
                    AssignProcessToJobObject(static_cast<HANDLE>(windowsJobHandle_), hProc);
                    CloseHandle(hProc);
                }
            }
        }
#endif
        sendConfigureRequest();
    });
    connect(
        daemonProcess_,
        qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
        this,
        [this](int exitCode, QProcess::ExitStatus) {
            endFullStream();
            daemonConfigured_ = false;
            checkpointLoadedInDaemon_ = false;
            activeRequestId_ = 0;
            pendingBindSongId_ = 0;
            boundSongKey_.clear();
            if (enabled_) {
                updateStatus(
                    QStringLiteral("error"),
                    QStringLiteral("进程已退出 (%1)").arg(exitCode),
                    statusDetail_);
            }
        });

    updateStatus(QStringLiteral("starting"), QStringLiteral("正在启动..."));
    const QStringList args{QStringLiteral("-u"), QStringLiteral("-m"), QStringLiteral("maidata.spine.copilot")};
    connect(daemonProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart)
            updateStatus(QStringLiteral("error"), QStringLiteral("无法启动推理环境"), daemonProcess_->errorString());
    });
    if (usesWsl()) {
        QStringList launch;
        if (!wslDistribution_.isEmpty()) launch << QStringLiteral("--distribution") << wslDistribution_;
        launch << QStringLiteral("--cd") << QDir::toNativeSeparators(sourceRoot)
               << QStringLiteral("--exec") << QStringLiteral("env") << QStringLiteral("PYTHONIOENCODING=utf-8")
               << QStringLiteral("PYTHONNOUSERSITE=1") << ("OMP_NUM_THREADS=" + QString::number(cpuThreads_)) << pythonExe;
        daemonProcess_->start(QStringLiteral("wsl.exe"), launch + args);
    } else {
        daemonProcess_->start(pythonExe, args);
    }
}

void NoaCopilotModel::stopDaemon()
{
    endFullStream();
    stdoutBuffer_.clear();
    warmTimer_.stop();
    debounceTimer_.stop();
    idleVramTimer_.stop();
    activeRequestId_ = 0;
    pendingBindSongId_ = 0;
    daemonConfigured_ = false;
    checkpointLoadedInDaemon_ = false;
    boundSongKey_.clear();

    if (daemonProcess_ == nullptr) {
        return;
    }
    disconnect(daemonProcess_, nullptr, this, nullptr);
    if (daemonProcess_->state() != QProcess::NotRunning) {
        QJsonObject closeCmd;
        closeCmd.insert(QStringLiteral("op"), QStringLiteral("close"));
        writeDaemonJson(closeCmd);
        if (!daemonProcess_->waitForFinished(500)) {
            daemonProcess_->kill();
            daemonProcess_->waitForFinished(500);
        }
    }
    daemonProcess_->deleteLater();
    daemonProcess_ = nullptr;
}

void NoaCopilotModel::writeDaemonJson(const QJsonObject& payload)
{
    if (daemonProcess_ == nullptr || daemonProcess_->state() != QProcess::Running) {
        return;
    }
    const QByteArray line = QJsonDocument(payload).toJson(QJsonDocument::Compact) + '\n';
    daemonProcess_->write(line);
}

void NoaCopilotModel::sendConfigureRequest()
{
    if (daemonProcess_ == nullptr || daemonProcess_->state() != QProcess::Running) {
        return;
    }
    updateStatus(
        hasCheckpoint() ? QStringLiteral("starting") : QStringLiteral("missing_checkpoint"),
        hasCheckpoint() ? QStringLiteral("正在加载权重...") : QStringLiteral("未选择权重"));
    QJsonObject req;
    req.insert("cpu_threads", cpuThreads_);
    req.insert("mert_device", mertDevice_);
    QString python = mertDevice_ == "cuda" ? cudaPython_ : cpuPython_;
    if (python.isEmpty()) python = QDir(defaultBuiltinRoot()).filePath("env/python.exe");
    QJsonArray command;
    if (python != resolvedPythonPath() || mertDevice_ != device_) {
        if (python.startsWith('/') && !usesWsl()) {
            command.append("wsl.exe");
            if (!wslDistribution_.isEmpty()) { command.append("-d"); command.append(wslDistribution_); }
        } else command.append(python);
    }
    req["mert_command"] = command;
    req["mert_python"] = python;
    req.insert(QStringLiteral("op"), QStringLiteral("configure"));
    req.insert(QStringLiteral("id"), nextRpcId_++);
    req.insert(QStringLiteral("checkpoint"), checkpointPath_);
    req.insert(QStringLiteral("device"), device_);
    req.insert(QStringLiteral("precision"), precision_);
    req.insert(QStringLiteral("models"), QDir(defaultBuiltinRoot()).filePath(QStringLiteral("models")));
    req.insert(QStringLiteral("cache_dir"), resolvedCacheDir());
    writeDaemonJson(req);
}

void NoaCopilotModel::sendBindSongIfNeeded()
{
    if (!enabled_ || !document_.hasDocument()) {
        return;
    }
    const QString audioPath = miacode::chart_assets::resolveTrackPath(document_.currentFilePath());
    const QFileInfo audioInfo(audioPath);
    if (audioPath.isEmpty() || !audioInfo.exists() || !audioInfo.isFile()) {
        return;
    }
    if (daemonProcess_ == nullptr || daemonProcess_->state() == QProcess::NotRunning) {
        ensureDaemonRunning();
        return;
    }
    if (!daemonConfigured_) {
        return;
    }
    const QString songKey = QStringLiteral("%1|%2|%3|%4")
                                .arg(QDir::cleanPath(audioInfo.absoluteFilePath()))
                                .arg(audioInfo.size())
                                .arg(audioInfo.lastModified().toMSecsSinceEpoch())
                                .arg(checkpointLoadedInDaemon_ ? checkpointPath_ : QString());
    if (boundSongKey_ == songKey) {
        return;
    }
    if (pendingBindSongId_ > 0) {
        QJsonObject cancelReq;
        cancelReq.insert(QStringLiteral("op"), QStringLiteral("cancel"));
        cancelReq.insert(QStringLiteral("id"), pendingBindSongId_);
        writeDaemonJson(cancelReq);
        pendingBindSongId_ = 0;
    }
    boundSongKey_ = songKey;
    const int reqId = nextRpcId_++;
    pendingBindSongId_ = reqId;
    QJsonObject req;
    req = chartRequest();
    req.insert(QStringLiteral("op"), QStringLiteral("bind_song"));
    req.insert(QStringLiteral("id"), reqId);
    warmTimer_.start();
    req.insert(QStringLiteral("audio"), audioPath);
    req.insert(QStringLiteral("first"), currentChartFirstSeconds());
    req.insert(QStringLiteral("bpm"), currentChartBpm());
    req.insert(QStringLiteral("text"), document_.chartText());
    writeDaemonJson(req);
}

void NoaCopilotModel::sendSyncChartIfNeeded()
{
    if (!enabled_ || !daemonConfigured_ || !document_.hasDocument() || document_.currentDifficultyId() <= 0) {
        return;
    }
    const quint64 rev = document_.documentRevision();
    if (syncedChartRevision_ == rev) {
        return;
    }
    syncedChartRevision_ = rev;
    QJsonObject req;
    req.insert(QStringLiteral("op"), QStringLiteral("sync_chart"));
    req.insert(QStringLiteral("id"), nextRpcId_++);
    req.insert(QStringLiteral("revision"), static_cast<qint64>(rev));
    req.insert(QStringLiteral("text"), document_.chartText());
    req.insert(QStringLiteral("first"), currentChartFirstSeconds());
    req.insert(QStringLiteral("bpm"), currentChartBpm());
    req.insert(QStringLiteral("difficulty"), document_.currentDifficultyId());
    req.insert(QStringLiteral("level"), parseLevelNumeric(document_.currentDifficultyLevel()));
    req.insert(QStringLiteral("designer"), effectiveDesigner());
    writeDaemonJson(req);
}

void NoaCopilotModel::sendSuggestRequest(
    const QString& mode,
    int spanBeats,
    bool fillToNext,
    const QString& levelOverride,
    const QString& designerOverride)
{
    if (!enabled_) {
        return;
    }
    ensureDaemonRunning();
    if (!hasCheckpoint()) {
        updateStatus(
            QStringLiteral("missing_checkpoint"),
            QStringLiteral("未选择权重"),
            QStringLiteral("点击状态栏 NoaCopilot -> 权重目录，放入 .ckpt 文件即可"));
        return;
    }
    if (daemonProcess_ == nullptr || daemonProcess_->state() != QProcess::Running) {
        return;
    }

    cancelActiveGeneration();
    dismissGhost();
    discardCandidates();
    pendingAcceptAfterStream_ = false;
    idleVramTimer_.start();

    const int reqId = nextRpcId_++;
    activeRequestId_ = reqId;
    activeRequestMode_ = mode;
    activeRequestRevision_ = document_.documentRevision();

    if (mode == QStringLiteral("full")) {
        fullStreamingActive_ = true;
        streamingProgress_ = 0.0;
        streamingBeat_ = 0;
        streamingTotalBeats_ = 0;
        lastStreamFlush_.start();
        emit streamingProgressChanged();
        emit emptyCardChanged();
        if (statusState_ != QStringLiteral("warming_audio")) {
            updateStatus(QStringLiteral("generating"), QStringLiteral("正在流式生成整曲..."));
        }
    } else if (mode == QStringLiteral("range")) {
        if (statusState_ != QStringLiteral("warming_audio")) {
            updateStatus(QStringLiteral("generating"), QStringLiteral("正在生成段落..."));
        }
    } else {
        if (statusState_ != QStringLiteral("warming_audio")) {
            updateStatus(QStringLiteral("generating"), QStringLiteral("正在续写..."));
        }
    }

    const QString audioPath = miacode::chart_assets::resolveTrackPath(document_.currentFilePath());
    const double levelVal = !levelOverride.trimmed().isEmpty()
        ? parseLevelNumeric(levelOverride)
        : parseLevelNumeric(document_.currentDifficultyLevel());
    const QString designerVal = effectiveDesigner(designerOverride);

    QJsonObject req;
    req = chartRequest();
    req.insert(QStringLiteral("op"), QStringLiteral("suggest"));
    req.insert(QStringLiteral("id"), reqId);
    req.insert(QStringLiteral("revision"), static_cast<qint64>(activeRequestRevision_));
    req.insert(QStringLiteral("mode"), mode);
    req.insert(QStringLiteral("text"), document_.chartText());
    req.insert(QStringLiteral("audio"), audioPath);
    req.insert(QStringLiteral("first"), currentChartFirstSeconds());
    req.insert(QStringLiteral("bpm"), currentChartBpm());
    req.insert(QStringLiteral("difficulty"), document_.currentDifficultyId());
    req.insert(QStringLiteral("level"), levelVal);
    req.insert(QStringLiteral("designer"), designerVal);
    req.insert(QStringLiteral("cursor_offset"), lastCursorPos_);
    req.insert(QStringLiteral("sel_start"), lastSelStart_);
    req.insert(QStringLiteral("sel_end"), lastSelEnd_);
    req.insert(QStringLiteral("span_beats"), qMax(1, spanBeats));
    req.insert(QStringLiteral("fill_to_next"), fillToNext);
    req.insert(QStringLiteral("temperature"), temperature_);
    req.insert(QStringLiteral("slide_bias"), slideBias_);
    req.insert(QStringLiteral("seed"), seed_);
    request_ = req;
    writeDaemonJson(req);
}

void NoaCopilotModel::handleDaemonReadyReadStdout()
{
    if (daemonProcess_ == nullptr) {
        return;
    }
    stdoutBuffer_.append(daemonProcess_->readAllStandardOutput());
    while (true) {
        const int newlineIdx = stdoutBuffer_.indexOf('\n');
        if (newlineIdx < 0) {
            break;
        }
        const QByteArray line = stdoutBuffer_.left(newlineIdx).trimmed();
        stdoutBuffer_.remove(0, newlineIdx + 1);
        if (line.isEmpty()) {
            continue;
        }
        QJsonParseError err;
        const QJsonDocument doc = QJsonDocument::fromJson(line, &err);
        if (err.error == QJsonParseError::NoError && doc.isObject()) {
            handleDaemonMessage(doc.object());
        }
    }
}

void NoaCopilotModel::handleDaemonReadyReadStderr()
{
    if (daemonProcess_ == nullptr) {
        return;
    }
    const QString errText = QString::fromUtf8(daemonProcess_->readAllStandardError()).trimmed();
    if (!errText.isEmpty()) {
        miacode::debug_log::appendLine(
            miacode::debug_log::Channel::Runtime,
            QStringLiteral("noa_copilot/stderr"),
            errText);
        statusDetail_ = errText.section(QLatin1Char('\n'), -1);
        emit statusChanged();
    }
}

QJsonObject NoaCopilotModel::chartRequest() const
{
    const QString extra = document_.metadataExtraText();
    QJsonObject request{{"protocol",2},{"audio",miacode::chart_assets::resolveTrackPath(document_.currentFilePath())},
        {"text",document_.chartText()},{"first",currentChartFirstSeconds()},{"bpm",currentChartBpm()},
        {"level",parseLevelNumeric(document_.currentDifficultyLevel())},{"difficulty",document_.currentDifficultyId()},
        {"designer",effectiveDesigner()},{"line_align_beats",lineAlignBeats_},{"candidate_count",candidateCount_},
        {"mert_cache",mertCache_},{"roformer_cache",roformerCache_}};
    for (const auto& key : {QStringLiteral("shortid"),QStringLiteral("cabinet"),QStringLiteral("version")}) {
        QRegularExpression field("^&" + key + "=([^\\r\\n]+)",QRegularExpression::MultilineOption);
        const auto match = field.match(extra);
        if (match.hasMatch()) request[key=="shortid" ? "song_id" : key] = match.captured(1).trimmed();
    }
    return request;
}

void NoaCopilotModel::scanEnvironments(bool discover)
{
    const int generation=++scanGeneration_;
    const QString cpu=cpuPython_.isEmpty() ? QDir(defaultBuiltinRoot()).filePath("env/python.exe") : cpuPython_;
    const QString cuda=cudaPython_, distro=wslDistribution_;
    environmentStatus_=QStringLiteral("正在检测 PyTorch 环境…");
    emit configChanged();
    auto rows=std::make_shared<QVariantList>();
    QThread* thread=QThread::create([cpu,cuda,distro,discover,rows] {
        *rows=probeNoaEnvironments(cpu,cuda,distro,discover);
    });
    connect(thread,&QThread::finished,thread,&QObject::deleteLater);
    connect(thread,&QThread::finished,this,[this,generation,cpu,cuda,rows] {
        if (generation!=scanGeneration_) return;
        environments_=*rows;
        cpuAvailable_=cudaAvailable_=false;
        QString reason;
        for (const auto& v : *rows) {
            const auto row=v.toMap();
            const QString path=row["path"].toString();
            if (path==QDir::fromNativeSeparators(cpu)) {
                cpuAvailable_ = row["cpu"].toBool();
                if (!cpuAvailable_) reason = row["reason"].toString();
            }
            if (path==QDir::fromNativeSeparators(cuda)) {
                cudaAvailable_=row["cuda"].toBool();
                if (!cudaAvailable_) reason=row["reason"].toString();
            }
        }
        environmentStatus_=reason;
        emit configChanged();
    });
    thread->start();
}

void NoaCopilotModel::warmSong()
{
    if (!enabled_ || !daemonConfigured_ || !hasCheckpoint() || fullStreamingActive_ || !document_.hasDocument()) return;
    auto req = chartRequest();
    if (req["audio"].toString().isEmpty()) return;
    static const QRegularExpression bpmTag(R"(\(\s*[1-9]\d*(?:\.\d+)?\s*\))");
    req["valid_bpm"] = document_.wholeBpm().toDouble() > 0 || bpmTag.match(document_.chartText()).hasMatch();
    req["op"] = "warm";
    req["id"] = nextRpcId_++;
    writeDaemonJson(req);
}

bool NoaCopilotModel::hasSkeleton(bool selectionOnly) const
{
    if (document_.wholeSourceEditorActive() || !document_.hasDocument()) return false;
    const QString body = document_.chartText();
    const int lo = qMin(lastSelStart_,lastSelEnd_), hi = qMax(lastSelStart_,lastSelEnd_);
    const bool selected = lo!=hi;
    if (selectionOnly && !selected) return false;
    const int pos = selected ? lo : lastCursorPos_;
    const int a = pos > 0 ? int(body.lastIndexOf('\n',pos-1))+1 : 0;
    const int nl = body.indexOf('\n',selected ? hi : pos);
    const int b = nl < 0 ? int(body.size()) : nl;
    static const QRegularExpression item(R"((?:^|[,/})])\s*(x|x?\*+)\s*(?=[/,]|$))",
        QRegularExpression::CaseInsensitiveOption | QRegularExpression::MultilineOption);
    QString chunk = body.mid(a,b-a);
    static const QRegularExpression comment(R"(\|\|[^\n]*)");
    auto comments = comment.globalMatch(chunk);
    while (comments.hasNext()) { const auto m = comments.next(); chunk.replace(m.capturedStart(),m.capturedLength(),QString(m.capturedLength(),' ')); }
    auto matches = item.globalMatch(chunk);
    while (matches.hasNext()) {
        const auto match = matches.next();
        if (!selected || (a + match.capturedStart(1) < hi && a + match.capturedEnd(1) > lo)) return true;
    }
    return false;
}

void NoaCopilotModel::mouseSkeletonSelection(bool quick)
{
    skeletonSelection_ = enabled_ && hasSkeleton(true);
    emit ghostStateChanged();
    if (skeletonSelection_ && quick) triggerManualContinue();
}

QString NoaCopilotModel::candidateHint() const
{
    if (!selected_) return QStringLiteral("原谱 #0");
    QString label=QStringLiteral("候选 %1/%2").arg(selected_).arg(exhausted_ ? highCandidate_ : qMax(candidateCount_,highCandidate_));
    const auto it=candidates_.constFind(selected_);
    if (it!=candidates_.cend() && it->localIssues) label+=QStringLiteral(" · 局部无理 %1").arg(it->localIssues);
    return label;
}

void NoaCopilotModel::discardCandidates()
{
    renderTimer_.stop();
    strikeLeft_ = strikeRight_ = -1;
    if (!candidates_.isEmpty()) {
        preview_.setCopilotReview(false);
        if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) bridge->clearGhostRangeCache();
        if (activeRequestId_) writeDaemonJson({{"op","cancel"},{"id",activeRequestId_}});
        writeDaemonJson({{"op","discard"},{"id",nextRpcId_++}});
        activeRequestId_ = 0;
    }
    selected_ = wanted_ = highCandidate_ = 0;
    exhausted_ = false;
    candidates_.clear();
    stepRequest_ = {};
}

void NoaCopilotModel::requestMore()
{
    if (activeRequestId_ || exhausted_ || !highCandidate_) return;
    activeRequestId_ = nextRpcId_++;
    QJsonObject req = request_;
    req["op"] = "more";
    req["id"] = activeRequestId_;
    writeDaemonJson(req);
}

void NoaCopilotModel::switchCandidate(int direction)
{
    if (candidates_.isEmpty()) return;
    int next = selected_+direction;
    if (direction<0) {
        if (next<0) return;
        while (next > 0 && !candidates_.contains(next)) --next;
    } else {
        while (next <= highCandidate_ && !candidates_.contains(next)) ++next;
    }
    if (direction > 0 && (!candidates_.contains(next) || !candidates_[next].ready)) {
        if (exhausted_ && next>highCandidate_) {
            updateStatus("ready",QStringLiteral("诺亚想不出来更多的写法了哦"));
            emit ghostStateChanged();
            return;
        }
        wanted_ = next;
        requestMore();
        return;
    }
    wanted_=0;
    selected_=next;
    writeDaemonJson({{"op","select"},{"id",nextRpcId_++},{"candidate",next}});
    showCandidate();
    if (direction>0 && next==highCandidate_) requestMore();
}

void NoaCopilotModel::prepareCandidatePreview(Candidate& candidate, const QString& simai)
{
    if (candidate.previewReady && candidate.previewKey == simai) return;
    compileGhost(simai, candidate.startSecond, candidate.activeBpm, candidate.markers, candidate.timeline);
    candidate.localIssues = preview_.localGhostIssues(candidate.markers, candidate.startSecond,
        candidate.writer.layout()["end_second"].toDouble());
    candidate.previewKey = simai;
    candidate.previewReady = true;
}

void NoaCopilotModel::showCandidate()
{
    if (candidates_.isEmpty()) return;
    if (!selected_) {
        ghostActive_ = true;
        ghostStreaming_ = false;
        ghostText_.clear();
        ghostReplaceStart_ = ghostReplaceEnd_ = lastCursorPos_;
        emit ghostStateChanged();
        updateEditorGhostStrike();
        clearDualSurfaceGhostPreview();
        return;
    }
    auto& c = candidates_[selected_];
    const auto& w = c.writer;
    if (fullStreamingActive_) {
        publishFullChart(w.text());
        return;
    }
    ghostActive_ = true;
    ghostStreaming_ = !c.ready;
    ghostText_ = c.stepsTaken ? c.remainder : w.text();
    ghostReplaceStart_=c.left; ghostReplaceEnd_=c.right;
    ghostOrigSpanEnd_=c.origEnd; ghostOrigCut_=c.cut; ghostBridgeLen_=c.bridgeLength;
    ghostCurrentStep_=c.stepsTaken;
    const QString preview = c.stepsTaken ? c.remainingPreview : w.preview();
    updateGhostSplitLines();
    emit ghostStateChanged();
    updateEditorGhostStrike();
    applyDualSurfaceGhostPreview(preview,c.startSecond,c.activeBpm,c.left,c.right);
}

bool NoaCopilotModel::handleStructured(const QJsonObject& m)
{
    const QString event=m["event"].toString();
    if (event!="candidate_begin" && event!="moment" && event!="beat" && event!="candidate_end" && event!="pool_end")
        return false;
    if (m["id"].toInt()!=activeRequestId_ || activeRequestRevision_!=document_.documentRevision()) return true;
    const int number=m["candidate"].toInt();
    if (event=="candidate_begin") {
        preview_.setCopilotReview(true);
        auto& c=candidates_[number];
        c = Candidate();
        c.writer.begin(m,document_.chartText(),lineAlignBeats_);
        c.left=c.cut=m["left"].toInt(); c.right=c.origEnd=m["right"].toInt();
        c.startSecond=m["start_second"].toDouble(); c.activeBpm=c.writer.bpm(m["start"].toInt());
        if (!highCandidate_ && number==1) selected_=1;
        if (inlineCardVisible_) closeInlineCard();
        return true;
    }
    if (event=="pool_end") {
        activeRequestId_=0;
        exhausted_=m["exhausted"].toBool();
        if (!highCandidate_ && exhausted_) {
            dismissGhost();
            updateStatus("ready",QStringLiteral("诺亚想不出来更多的写法了哦"));
        } else if (wanted_>highCandidate_ && !exhausted_) {
            requestMore();
        } else if (wanted_>highCandidate_ && exhausted_) {
            wanted_=0;
            updateStatus("ready",QStringLiteral("诺亚想不出来更多的写法了哦"));
        } else {
            updateStatus("ready",QStringLiteral("就绪"),activeCheckpointName());
        }
        emit ghostStateChanged();
        return true;
    }
    if (!candidates_.contains(number)) return true;
    auto& c=candidates_[number];
    if (event=="candidate_end") {
        c.writer.finish();
        c.ready=true;
        if (!fullStreamingActive_) prepareCandidatePreview(c, c.writer.preview());
        highCandidate_=qMax(number,highCandidate_);
        if (wanted_==number) { selected_=number; wanted_=0; }
        if (number==selected_) {
            renderTimer_.stop();
            showCandidate();
            if (fullStreamingActive_) {
                streamingBeat_ = streamingTotalBeats_;
                streamingProgress_ = 1.0;
                emit streamingProgressChanged();
                endFullStream();
                discardCandidates();
                updateStatus("ready",QStringLiteral("整曲草稿已生成"),activeCheckpointName());
            } else if (pendingAcceptAfterStream_) {
                acceptFullGhost();
            }
        }
        while (candidates_.size()>8) {
            auto oldest=candidates_.begin();
            while (oldest.key() == 1 || oldest.key() == selected_) ++oldest;
            candidates_.erase(oldest);
        }
    } else {
        if (event=="moment") c.writer.moment(m["tick"].toInt(),m["notes"].toArray());
        else c.writer.advance(m["tick"].toInt());
        if (number==selected_ && (fullStreamingActive_ || !c.ready)) {
            if (fullStreamingActive_) {
                streamingBeat_=c.writer.position()/96;
                streamingTotalBeats_=(c.writer.layout()["stop"].toInt()+95)/96;
                streamingProgress_=double(streamingBeat_)/qMax(1,streamingTotalBeats_);
                emit streamingProgressChanged();
            }
            if (!renderTimer_.isActive()) renderTimer_.start();
        }
    }
    return true;
}

void NoaCopilotModel::handleDaemonMessage(const QJsonObject& message)
{
    if (handleStructured(message)) return;
    const QString event = message.value(QStringLiteral("event")).toString();
    if (event == QStringLiteral("configured")) {
        daemonConfigured_ = message.value(QStringLiteral("ok")).toBool();
        checkpointLoadedInDaemon_ = message.value(QStringLiteral("checkpoint_loaded")).toBool();
        const QString dev = message.value(QStringLiteral("device")).toString();
        const QString prec = message.value(QStringLiteral("precision")).toString();
        const QString repr = message.value(QStringLiteral("representation")).toString();
        if (!dev.isEmpty()) {
            deviceInfo_ = QStringLiteral("%1 · %2").arg(dev.toUpper(), prec.toUpper());
            if (!repr.isEmpty()) {
                deviceInfo_ += QStringLiteral(" · %1").arg(repr);
            }
        }
        QStringList vocab;
        const QJsonArray arr = message.value(QStringLiteral("designer_vocab")).toArray();
        for (const QJsonValue& val : arr) {
            vocab.append(val.toString());
        }
        if (!vocab.isEmpty()) {
            rebuildDesignerOptions(vocab);
        }
        const QString err = message.value(QStringLiteral("error")).toString();
        if (!err.isEmpty()) {
            updateStatus(QStringLiteral("error"), QStringLiteral("权重加载失败"), err);
        } else if (checkpointLoadedInDaemon_) {
            updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
            sendBindSongIfNeeded();
        } else {
            updateStatus(
                QStringLiteral("missing_checkpoint"),
                QStringLiteral("未加载权重"),
                QStringLiteral("请放入或选择模型权重文件"));
            if (daemonConfigured_) {
                sendBindSongIfNeeded();
            }
        }
        return;
    }

    if (event == QStringLiteral("status")) {
        QString st = message.value(QStringLiteral("state")).toString(QStringLiteral("warming_audio"));
        QString msg = message.value(QStringLiteral("message")).toString(QStringLiteral("正在处理..."));
        if (st == QStringLiteral("encoding_audio")) {
            st = QStringLiteral("warming_audio");
            msg = QStringLiteral("正在分析音频...");
        }
        updateStatus(st, msg);
        return;
    }

    if (event == QStringLiteral("song_bound")) {
        const int id = message.value(QStringLiteral("id")).toInt();
        if (id == pendingBindSongId_ || id <= 0) {
            pendingBindSongId_ = 0;
        }
        refreshAudioCacheSize();
        if (activeRequestId_ > 0) {
            if (statusState_ == QStringLiteral("warming_audio")) {
                const QString genLabel = activeRequestMode_ == QStringLiteral("full")
                    ? QStringLiteral("正在流式生成整曲...")
                    : (activeRequestMode_ == QStringLiteral("range")
                           ? QStringLiteral("正在生成段落...")
                           : QStringLiteral("正在续写..."));
                updateStatus(QStringLiteral("generating"), genLabel);
            }
        } else if (checkpointLoadedInDaemon_) {
            updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
        } else {
            updateStatus(
                QStringLiteral("missing_checkpoint"),
                QStringLiteral("未加载权重"),
                QStringLiteral("请放入或选择模型权重文件"));
        }
        return;
    }

    if (event == QStringLiteral("stream_chunk")) {
        const int id = message.value(QStringLiteral("id")).toInt();
        if (id != activeRequestId_) {
            return;
        }
        const QString mode = message.value(QStringLiteral("mode")).toString(activeRequestMode_);
        streamingBeat_ = message.value(QStringLiteral("beat")).toInt();
        streamingTotalBeats_ = message.value(QStringLiteral("total_beats")).toInt();
        streamingProgress_ = message.value(QStringLiteral("progress")).toDouble();
        emit streamingProgressChanged();

        if (mode == QStringLiteral("full")) {
            updateStatus(
                QStringLiteral("generating"),
                QStringLiteral("整曲生成 (%1/%2 拍)").arg(streamingBeat_).arg(streamingTotalBeats_));

            const QString chart = message.value(QStringLiteral("chart_text")).toString();
            if (fullStreamingActive_ && !chart.isEmpty()) {
                pendingStreamChart_ = chart;
                if (!streamFlushTimer_.isActive())
                    streamFlushTimer_.start(int(qMax<qint64>(0, 2000 - lastStreamFlush_.elapsed())));
            } else if (const QString ghost = message.value(QStringLiteral("ghost_simai")).toString();
                       fullStreamingActive_ && !ghost.isEmpty() && lastStreamFlush_.elapsed() >= 2000) {
                // Older custom daemons only publish a short ghost.
                lastStreamFlush_.restart();
                applyDualSurfaceGhostPreview(ghost, message.value(QStringLiteral("start_second")).toDouble(currentChartFirstSeconds()),
                                             message.value(QStringLiteral("active_bpm")).toDouble(currentChartBpm()));
            }
            return;
        }

        if (activeRequestRevision_ != document_.documentRevision()) {
            return;
        }
        const QString text = message.value(QStringLiteral("text")).toString();
        if (text.isEmpty()) {
            return;
        }
        const int replaceStart = message.value(QStringLiteral("replace_start")).toInt(lastCursorPos_);
        const int replaceEnd = message.value(QStringLiteral("replace_end")).toInt(lastCursorPos_);
        const double startSec = message.value(QStringLiteral("start_second")).toDouble(currentChartFirstSeconds());
        const double endSec = message.value(QStringLiteral("end_second")).toDouble(startSec + 2.0);
        const double activeBpm = message.value(QStringLiteral("active_bpm")).toDouble(currentChartBpm());
        const QString ghostSimai = message.value(QStringLiteral("ghost_simai")).toString();
        const int noteCount = message.value(QStringLiteral("note_count")).toInt();

        if (mode == QStringLiteral("range")) {
            if (inlineCardVisible_) {
                inlineCardVisible_ = false;
                emit inlineCardChanged();
            }
            pendingRangeActive_ = true;
            rangeStreaming_ = true;
            pendingRangeText_ = text;
            pendingReplaceStart_ = replaceStart;
            pendingReplaceEnd_ = replaceEnd;
            pendingStartSecond_ = startSec;
            pendingEndSecond_ = endSec;
            pendingNoteCount_ = noteCount;
            emit pendingRangeChanged();
            updateEditorGhostStrike();
            applyDualSurfaceGhostPreview(ghostSimai, startSec, activeBpm, replaceStart, replaceEnd);
            return;
        }

        // mode == "continue" streaming
        ghostActive_ = true;
        ghostStreaming_ = true;
        ghostText_ = text;
        ghostReplaceStart_ = replaceStart;
        ghostReplaceEnd_ = replaceEnd;
        ghostOrigSpanEnd_ = replaceEnd;
        ghostOrigCut_ = replaceStart;
        ghostBridgeLen_ = 0;
        ghostSteps_ = QJsonArray();
        ghostCurrentStep_ = 0;
        updateGhostSplitLines();
        emit ghostStateChanged();
        updateEditorGhostStrike();
        applyDualSurfaceGhostPreview(ghostSimai, startSec, activeBpm, replaceStart, replaceEnd);
        return;
    }

    if (event == QStringLiteral("result")) {
        const int id = message.value(QStringLiteral("id")).toInt();
        if (id != activeRequestId_) {
            return;
        }
        streamFlushTimer_.stop();
        pendingStreamChart_.clear();
        const bool ok = message.value(QStringLiteral("ok")).toBool();
        const QString mode = message.value(QStringLiteral("mode")).toString();
        const QString err = message.value(QStringLiteral("error")).toString();
        if (!ok) {
            activeRequestId_ = 0;
            pendingAcceptAfterStream_ = false;
            if (ghostStreaming_) {
                ghostStreaming_ = false;
                dismissGhost();
            }
            if (rangeStreaming_) {
                rangeStreaming_ = false;
                rejectPendingRange();
            }
            endFullStream();
            updateStatus(QStringLiteral("error"), QStringLiteral("生成失败"), err);
            return;
        }

        updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
        refreshAudioCacheSize();

        if (activeRequestRevision_ != document_.documentRevision()) {
            activeRequestId_ = 0;
            pendingAcceptAfterStream_ = false;
            if (ghostStreaming_) {
                ghostStreaming_ = false;
                dismissGhost();
            }
            if (rangeStreaming_) {
                rangeStreaming_ = false;
                rejectPendingRange();
            }
            endFullStream();
            return;
        }
        const QString text = message.value(QStringLiteral("text")).toString();
        const int replaceStart = message.value(QStringLiteral("replace_start")).toInt(lastCursorPos_);
        const int replaceEnd = message.value(QStringLiteral("replace_end")).toInt(lastCursorPos_);
        const double startSec = message.value(QStringLiteral("start_second")).toDouble(currentChartFirstSeconds());
        const double endSec = message.value(QStringLiteral("end_second")).toDouble(startSec + 2.0);
        const double activeBpm = message.value(QStringLiteral("active_bpm")).toDouble(currentChartBpm());
        const QString ghostSimai = message.value(QStringLiteral("ghost_simai")).toString();
        const int noteCount = message.value(QStringLiteral("note_count")).toInt();

        if (mode == QStringLiteral("full")) {
            if (publishFullChart(text)) streamingProgress_ = 1.0;
            activeRequestId_ = 0;
            pendingAcceptAfterStream_ = false;
            endFullStream();
            return;
        }

        activeRequestId_ = 0;
        if (mode == QStringLiteral("range")) {
            rangeStreaming_ = false;
            if (text.trimmed().isEmpty()) {
                pendingAcceptAfterStream_ = false;
                if (pendingRangeActive_) rejectPendingRange();
                return;
            }
            inlineCardVisible_ = false;
            emit inlineCardChanged();

            pendingRangeActive_ = true;
            pendingRangeText_ = text;
            pendingReplaceStart_ = replaceStart;
            pendingReplaceEnd_ = replaceEnd;
            pendingStartSecond_ = startSec;
            pendingEndSecond_ = endSec;
            pendingNoteCount_ = noteCount;
            emit pendingRangeChanged();
            updateEditorGhostStrike();
            applyDualSurfaceGhostPreview(ghostSimai, startSec, activeBpm, replaceStart, replaceEnd);
            if (pendingAcceptAfterStream_) {
                pendingAcceptAfterStream_ = false;
                acceptPendingRange();
            }
            return;
        }

        // mode == "continue"
        ghostStreaming_ = false;
        if (text.trimmed().isEmpty()) {
            pendingAcceptAfterStream_ = false;
            if (ghostActive_) dismissGhost();
            return;
        }
        // Verify document hasn't moved away while the suggestion was in flight
        if (static_cast<quint64>(message.value(QStringLiteral("revision")).toInteger()) != document_.documentRevision()) {
            pendingAcceptAfterStream_ = false;
            if (ghostActive_) dismissGhost();
            return;
        }
        ghostActive_ = true;
        ghostText_ = text;
        ghostReplaceStart_ = replaceStart;
        ghostReplaceEnd_ = replaceEnd;
        ghostOrigSpanEnd_ = replaceEnd;
        ghostOrigCut_ = replaceStart;
        ghostBridgeLen_ = 0;
        ghostSteps_ = message.value(QStringLiteral("steps")).toArray();
        ghostCurrentStep_ = 0;
        updateGhostSplitLines();
        emit ghostStateChanged();
        updateEditorGhostStrike();
        applyDualSurfaceGhostPreview(ghostSimai, startSec, activeBpm, replaceStart, replaceEnd);
        if (pendingAcceptAfterStream_) {
            pendingAcceptAfterStream_ = false;
            acceptFullGhost();
        }
        return;
    }

    if (event == QStringLiteral("cancelled")) {
        const int id = message.value(QStringLiteral("id")).toInt();
        if (id == activeRequestId_ || id <= 0) {
            activeRequestId_ = 0;
            pendingAcceptAfterStream_ = false;
            if (ghostStreaming_) {
                ghostStreaming_ = false;
                dismissGhost();
            }
            if (rangeStreaming_) {
                rangeStreaming_ = false;
                rejectPendingRange();
            }
            endFullStream();
            if (checkpointLoadedInDaemon_) {
                updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
            }
        }
        return;
    }
}

bool NoaCopilotModel::publishFullChart(const QString& text)
{
    if (activeRequestRevision_ != document_.documentRevision()) {
        cancelActiveGeneration();
        return false;
    }
    lastStreamFlush_.restart();
    if (text == document_.chartText() && !document_.dirty()) return true;
    const bool persist = candidates_.isEmpty() || candidates_[selected_].ready || !diskFlush_.isValid() || diskFlush_.elapsed()>=1800;
    if (persist) diskFlush_.start();
    if (!document_.applyGeneratedChart(text,persist)) {
        cancelActiveGeneration();
        updateStatus(QStringLiteral("error"), QStringLiteral("无法保存生成中的谱面"));
        return false;
    }
    activeRequestRevision_ = document_.documentRevision();
    return true;
}

void NoaCopilotModel::endFullStream()
{
    streamFlushTimer_.stop();
    pendingStreamChart_.clear();
    if (!fullStreamingActive_) return;
    if (!candidates_.isEmpty() && activeRequestRevision_ == document_.documentRevision() && document_.dirty())
        document_.applyGeneratedChart(document_.chartText());
    fullStreamingActive_ = false;
    clearDualSurfaceGhostPreview();
    emit streamingProgressChanged();
    emit emptyCardChanged();
}

void NoaCopilotModel::updateGhostSplitLines()
{
    const int newlineIdx = ghostText_.indexOf(QLatin1Char('\n'));
    if (newlineIdx < 0) {
        ghostFirstLine_ = ghostText_;
        ghostExtraLines_.clear();
    } else {
        ghostFirstLine_ = ghostText_.left(newlineIdx);
        ghostExtraLines_ = ghostText_.mid(newlineIdx + 1);
    }
}

void NoaCopilotModel::applyDualSurfaceGhostPreview(
    const QString& ghostSimai,
    double startSecond,
    double activeBpm,
    int strikeStartUtf16,
    int strikeEndUtf16)
{
    const bool hasStrike = strikeEndUtf16 > strikeStartUtf16;
    if (ghostSimai.trimmed().isEmpty() && !hasStrike) {
        clearDualSurfaceGhostPreview();
        return;
    }

    int strikeStartLine = 0;
    int strikeStartCol = 0;
    int strikeEndLine = 0;
    int strikeEndCol = 0;
    if (hasStrike) {
        if (strikeRevision_ != document_.documentRevision() || strikeLeft_ != strikeStartUtf16 || strikeRight_ != strikeEndUtf16) {
            strikeRevision_ = document_.documentRevision();
            strikeLeft_ = strikeStartUtf16;
            strikeRight_ = strikeEndUtf16;
            const QString chart = document_.chartText();
            strikeFrom_ = utf16OffsetToLineCol(chart, strikeLeft_);
            strikeTo_ = utf16OffsetToLineCol(chart, strikeRight_);
        }
        const auto [sl, sc] = strikeFrom_;
        const auto [el, ec] = strikeTo_;
        strikeStartLine = sl;
        strikeStartCol = sc;
        strikeEndLine = el;
        strikeEndCol = ec;
    }

    QVector<TimelineNoteMarker> ghostMarkers;
    TimelineRenderSnapshot ghostTimelineSnapshot;
    Candidate* candidate = selected_ && candidates_.contains(selected_) ? &candidates_[selected_] : nullptr;
    if (candidate && candidate->ready) {
        prepareCandidatePreview(*candidate, ghostSimai);
        ghostMarkers = candidate->markers;
        ghostTimelineSnapshot = candidate->timeline;
    } else {
        compileGhost(ghostSimai, startSecond, activeBpm, ghostMarkers, ghostTimelineSnapshot);
    }

    // 1. Circular Preview + Playback SFX (excludes struck-through notes & gives ghost notes identical SFX)
    preview_.setGhostPreviewOverlay(
        ghostMarkers,
        hasStrike,
        strikeStartLine,
        strikeStartCol,
        strikeEndLine,
        strikeEndCol, candidate && !candidate->ready);

    // 2. Bottom Timeline Ghost Notes + Struck-through exclusion
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        bridge->setGhostStrikeRange(hasStrike, strikeStartLine, strikeStartCol, strikeEndLine, strikeEndCol);
        bridge->setGhostSnapshot(ghostTimelineSnapshot);
    }
}

void NoaCopilotModel::clearDualSurfaceGhostPreview()
{
    preview_.clearGhostPreviewOverlay();
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        bridge->setGhostStrikeRange(false, 0, 0, 0, 0);
        bridge->setGhostSnapshot({});
    }
}

bool NoaCopilotModel::applyEditorReplacement(int startPos, int endPos, const QString& text, int newCursorPos)
{
    if (editorBridge_ != nullptr) {
        bool ok = false;
        QMetaObject::invokeMethod(
            editorBridge_.data(),
            "replaceTextRange",
            Qt::DirectConnection,
            Q_RETURN_ARG(bool, ok),
            Q_ARG(int, startPos),
            Q_ARG(int, endPos),
            Q_ARG(QString, text),
            Q_ARG(int, newCursorPos));
        if (ok) {
            return true;
        }
    }
    QString current = document_.chartText();
    const int boundedStart = qBound(0, qMin(startPos, endPos), current.size());
    const int boundedEnd = qBound(boundedStart, qMax(startPos, endPos), current.size());
    current.replace(boundedStart, boundedEnd - boundedStart, text);
    document_.setChartText(current);
    return true;
}

void NoaCopilotModel::updateEditorGhostStrike()
{
    if (editorBridge_ == nullptr) {
        return;
    }
    bool active = false;
    int insertPos = 0;
    QString inlineText;
    int strikeStart = 0;
    int strikeEnd = 0;
    if (ghostActive_ && (!ghostText_.isEmpty() || ghostReplaceEnd_ > ghostReplaceStart_)) {
        active = true;
        insertPos = ghostReplaceStart_;
        inlineText = ghostText_;
        strikeStart = ghostReplaceStart_;
        strikeEnd = ghostReplaceEnd_;
    } else if (pendingRangeActive_ && (!pendingRangeText_.isEmpty() || pendingReplaceEnd_ > pendingReplaceStart_)) {
        active = true;
        insertPos = pendingReplaceStart_;
        inlineText = pendingRangeText_;
        strikeStart = pendingReplaceStart_;
        strikeEnd = pendingReplaceEnd_;
    }
    QMetaObject::invokeMethod(
        editorBridge_.data(),
        "setInlineGhostPreview",
        Qt::DirectConnection,
        Q_ARG(bool, active),
        Q_ARG(int, insertPos),
        Q_ARG(QString, inlineText),
        Q_ARG(int, strikeStart),
        Q_ARG(int, strikeEnd));
}

void NoaCopilotModel::bindEditor(QObject* editorObject)
{
    editorBridge_ = editorObject;
    updateEditorGhostStrike();
}

void NoaCopilotModel::notifyEditorCursorOrTextChanged(
    int cursorPos,
    int selStart,
    int selEnd,
    bool textChanged)
{
    if (acceptingStep_) return;
    const bool cursorMoved = (cursorPos != lastCursorPos_ || selStart != lastSelStart_ || selEnd != lastSelEnd_);
    lastCursorPos_ = cursorPos;
    lastSelStart_ = selStart;
    lastSelEnd_ = selEnd;
    idleVramTimer_.start();

    if ((textChanged || cursorMoved) && activeRequestId_ > 0
        && (activeRequestMode_ == QStringLiteral("continue") || activeRequestMode_ == QStringLiteral("range"))) {
        cancelActiveGeneration();
    }
    if (ghostActive_ && (textChanged || cursorMoved)) {
        dismissGhost();
    }
    const bool selection = selStart != selEnd && hasSkeleton(true);
    if (selection != skeletonSelection_) { skeletonSelection_ = selection; emit ghostStateChanged(); }
    if (selStart != selEnd) {
        debounceTimer_.stop();
    }
    if (textChanged && !fullStreamingActive_) {
        if (pendingRangeActive_) rejectPendingRange();
        sendSyncChartIfNeeded();
        if (enabled_ && autoContinue_ && hasCheckpoint() && !inlineCardVisible_ && !pendingRangeActive_ && selStart == selEnd) {
            debounceTimer_.start(debounceMs_);
        }
    }
}

void NoaCopilotModel::notifyEditorCursorOrTextChanged(
    int cursorPos,
    int selStart,
    int selEnd,
    const QString& text,
    bool imeComposing,
    bool completionActive,
    bool followPlaybackActive)
{
    const bool textChanged = (text != lastObservedEditorText_);
    lastObservedEditorText_ = text;
    if (imeComposing || completionActive || followPlaybackActive) {
        lastCursorPos_ = cursorPos;
        lastSelStart_ = selStart;
        lastSelEnd_ = selEnd;
        debounceTimer_.stop();
        if (activeRequestId_ > 0 && activeRequestMode_ == QStringLiteral("continue")) {
            cancelActiveGeneration();
        }
        if (ghostActive_) {
            dismissGhost();
        }
        return;
    }
    notifyEditorCursorOrTextChanged(cursorPos, selStart, selEnd, textChanged);
}

bool NoaCopilotModel::handleEditorKeyPress(int key, int modifiers)
{
    if (!enabled_) {
        return false;
    }
    const bool ctrl = (modifiers & Qt::ControlModifier) != 0;
    const bool shift = (modifiers & Qt::ShiftModifier) != 0;
    const bool alt = (modifiers & Qt::AltModifier) != 0;
    const bool meta = (modifiers & Qt::MetaModifier) != 0;
    const bool plainTab = (key == Qt::Key_Tab && !ctrl && !shift && !alt && !meta);

    if (!candidates_.isEmpty() && !fullStreamingActive_ && !ctrl && !alt && !shift && !meta) {
        if (key == Qt::Key_Up || key == Qt::Key_Down) {
            switchCandidate(key == Qt::Key_Up ? -1 : 1);
            return true;
        }
        if (key == Qt::Key_Right) {
            if (selected_ && !ghostStreaming_) acceptNextGhostStep();
            return true;
        }
        if (plainTab) {
            if (!selected_) dismissGhost();
            else if (ghostStreaming_) pendingAcceptAfterStream_ = true;
            else acceptFullGhost();
            return true;
        }
        if (key == Qt::Key_Escape) { dismissGhost(); return true; }
    }
    if (pendingRangeActive_) {
        if (plainTab) {
            if (rangeStreaming_) {
                pendingAcceptAfterStream_ = true;
                return true;
            }
            return acceptPendingRange();
        }
        if (key == Qt::Key_Escape) {
            if (activeRequestId_ > 0 && activeRequestMode_ == QStringLiteral("range")) {
                cancelActiveGeneration();
            }
            rejectPendingRange();
            return true;
        }
        if (!rangeStreaming_ && key == Qt::Key_R && !ctrl && !alt) {
            regeneratePendingRange();
            return true;
        }
        if (!rangeStreaming_ && key == Qt::Key_Space && !ctrl && !alt) {
            auditionPendingRange();
            return true;
        }
    }

    if (ghostActive_) {
        if (plainTab) {
            if (ghostStreaming_) {
                pendingAcceptAfterStream_ = true;
                return true;
            }
            return acceptFullGhost();
        }
        if (!ghostStreaming_ && key == Qt::Key_Right && ctrl && !shift && !alt) {
            return acceptNextGhostStep();
        }
        if (key == Qt::Key_Escape) {
            if (activeRequestId_ > 0 && activeRequestMode_ == QStringLiteral("continue")) {
                cancelActiveGeneration();
            }
            dismissGhost();
            return true;
        }
    }

    if (inlineCardVisible_) {
        if (key == Qt::Key_Escape) {
            closeInlineCard();
            return true;
        }
        if ((key == Qt::Key_Return || key == Qt::Key_Enter) && !ctrl && !shift && !alt) {
            triggerRangeGenerate();
            return true;
        }
    }

    if (key == Qt::Key_Escape && (fullStreamingActive_ || activeRequestId_ > 0)) {
        cancelActiveGeneration();
        return true;
    }

    // When NoaCopilot is enabled, plain Tab triggers continuation instead of
    // Scintilla's default tab/indent block fill; disabling NoaCopilot restores Tab.
    if (plainTab && !inlineCardVisible_ && !fullStreamingActive_
        && document_.hasDocument() && document_.currentDifficultyId() > 0
        && !document_.wholeSourceEditorActive()) {
        if (!busy()) {
            triggerManualContinue();
        }
        return true;
    }

    return false;
}

void NoaCopilotModel::triggerManualContinue()
{
    if (!enabled_ || !document_.hasDocument() || document_.currentDifficultyId() <= 0) {
        return;
    }
    debounceTimer_.stop();
    sendSuggestRequest(QStringLiteral("continue"), 4, false, QString(), QString());
}

bool NoaCopilotModel::acceptFullGhost()
{
    if (!candidates_.isEmpty() && !selected_) { dismissGhost(); return true; }
    if (ghostStreaming_) { pendingAcceptAfterStream_ = true; return true; }
    if (!ghostActive_ || ghostText_.isEmpty()) return false;
    const QString textToInsert = ghostText_;
    const int startPos = ghostReplaceStart_;
    const int endPos = ghostReplaceEnd_;
    dismissGhost();
    return applyEditorReplacement(startPos, endPos, textToInsert);
}

bool NoaCopilotModel::acceptNextGhostStep()
{
    if (!ghostActive_ || ghostStreaming_ || ghostText_.isEmpty()) {
        return false;
    }
    if (!candidates_.isEmpty() && selected_ && candidates_[selected_].ready) {
        const auto& writer = candidates_[selected_].writer;
        const auto step = writer.step(ghostCurrentStep_);
        if (step.isEmpty()) return acceptFullGhost();
        ghostSteps_ = QJsonArray{step,QJsonObject()};
        ghostCurrentStep_ = 0;
        stepRequest_ = request_;
        stepRequest_["step_tick"] = step["tick"];
        stepRequest_["candidate"] = selected_;
        stepRequest_["division"] = step["division"];
    }
    if (ghostSteps_.isEmpty() || ghostCurrentStep_ >= ghostSteps_.size()) {
        return acceptFullGhost();
    }

    const QJsonObject stepObj = ghostSteps_.at(ghostCurrentStep_).toObject();
    const QString stepText = stepObj.value(QStringLiteral("accept_text")).toString();
    const QString escapeBridge = stepObj.value(QStringLiteral("escape_bridge")).toString();
    const int stepOrigCut = stepObj.value(QStringLiteral("replace_end_u16")).toInt(ghostOrigCut_);
    const QString remText = stepObj.value(QStringLiteral("preview_remaining_text")).toString();
    const QString remGhostSimai = stepObj.value(QStringLiteral("remaining_ghost_simai")).toString();
    const double remStartSec = stepObj.value(QStringLiteral("remaining_start_second")).toDouble();
    const double remBpm = stepObj.value(QStringLiteral("remaining_bpm")).toDouble(currentChartBpm());

    if (stepText.isEmpty()) {
        return acceptFullGhost();
    }

    const int insertStart = ghostReplaceStart_;
    const bool hasMoreSteps = !remText.trimmed().isEmpty() && (ghostCurrentStep_ + 1 < ghostSteps_.size());
    const int extraOrig = qMax(0, stepOrigCut - ghostOrigCut_);
    const int consumeEnd = hasMoreSteps
        ? (insertStart + ghostBridgeLen_ + extraOrig)
        : ghostReplaceEnd_;
    const QString replacement = hasMoreSteps ? (stepText + escapeBridge) : stepText;
    const int nextCursor = insertStart + stepText.size();
    const QJsonArray savedSteps = ghostSteps_;
    const int nextStepIdx = ghostCurrentStep_ + 1;
    const int savedOrigSpanEnd = ghostOrigSpanEnd_;

    QScopedValueRollback guard(acceptingStep_,true);
    ghostActive_ = false;
    applyEditorReplacement(insertStart, consumeEnd, replacement, nextCursor);

    if (!hasMoreSteps) {
        dismissGhost();
        return true;
    }

    debounceTimer_.stop();
    ghostActive_ = true;
    ghostText_ = remText;
    ghostReplaceStart_ = nextCursor;
    ghostBridgeLen_ = escapeBridge.size();
    ghostOrigCut_ = qMax(ghostOrigCut_, stepOrigCut);
    ghostOrigSpanEnd_ = savedOrigSpanEnd;
    ghostReplaceEnd_ = ghostReplaceStart_ + ghostBridgeLen_ + qMax(0, ghostOrigSpanEnd_ - ghostOrigCut_);
    ghostSteps_ = savedSteps;
    ghostCurrentStep_ = nextStepIdx;
    updateGhostSplitLines();
    emit ghostStateChanged();
    updateEditorGhostStrike();
    applyDualSurfaceGhostPreview(remGhostSimai, remStartSec, remBpm, ghostReplaceStart_, ghostReplaceEnd_);
    if (!stepRequest_.isEmpty()) {
        const int original = stepRequest_["candidate"].toInt();
        auto candidate = candidates_[original];
        const int prior = candidate.stepsTaken;
        candidate.stepsTaken++;
        candidate.remainder = remText;
        candidate.remainingPreview = remGhostSimai;
        candidate.startSecond = remStartSec;
        candidate.activeBpm = remBpm;
        candidate.left = ghostReplaceStart_;
        candidate.right = ghostReplaceEnd_;
        candidate.origEnd = ghostOrigSpanEnd_;
        candidate.cut = ghostOrigCut_;
        candidate.bridgeLength = ghostBridgeLen_;
        if (activeRequestId_) writeDaemonJson({{"op","cancel"},{"id",activeRequestId_}});
        renderTimer_.stop();
        candidates_.clear();
        candidate.previewKey.clear();
        candidate.previewReady = false;
        candidates_.insert(1,candidate);
        selected_ = highCandidate_ = 1; wanted_ = 0; exhausted_ = false;
        activeRequestId_ = nextRpcId_++;
        activeRequestRevision_ = document_.documentRevision();
        lastCursorPos_ = lastSelStart_ = lastSelEnd_ = nextCursor;
        lastObservedEditorText_ = document_.chartText();
        stepRequest_["accepted_steps"] = prior+1;
        stepRequest_["id"] = activeRequestId_;
        stepRequest_["revision"] = qint64(activeRequestRevision_);
        stepRequest_["text"] = lastObservedEditorText_;
        stepRequest_["cursor_offset"] = nextCursor;
        stepRequest_["sel_start"] = stepRequest_["sel_end"] = nextCursor;
        stepRequest_["replace_start"] = nextCursor;
        stepRequest_["replace_end"] = ghostReplaceEnd_;
        request_ = stepRequest_;
        ghostCurrentStep_ = prior+1;
        writeDaemonJson(stepRequest_);
        stepRequest_ = {};
        emit ghostStateChanged();
    }
    return true;
}

void NoaCopilotModel::dismissGhost()
{
    discardCandidates();
    debounceTimer_.stop();
    pendingAcceptAfterStream_ = false;
    if (!ghostActive_ && !ghostStreaming_ && ghostText_.isEmpty() && ghostReplaceStart_ == 0 && ghostReplaceEnd_ == 0) {
        updateEditorGhostStrike();
        return;
    }
    ghostActive_ = false;
    ghostStreaming_ = false;
    ghostText_.clear();
    ghostFirstLine_.clear();
    ghostExtraLines_.clear();
    ghostReplaceStart_ = 0;
    ghostReplaceEnd_ = 0;
    ghostOrigSpanEnd_ = 0;
    ghostOrigCut_ = 0;
    ghostBridgeLen_ = 0;
    ghostSteps_ = QJsonArray();
    ghostCurrentStep_ = 0;
    emit ghostStateChanged();
    updateEditorGhostStrike();
    if (!pendingRangeActive_ && !fullStreamingActive_) {
        clearDualSurfaceGhostPreview();
    }
}

void NoaCopilotModel::dismissAllActivePreviews()
{
    if (activeRequestId_ > 0
        && (activeRequestMode_ == QStringLiteral("continue") || activeRequestMode_ == QStringLiteral("range"))) {
        cancelActiveGeneration();
    }
    if (ghostActive_ || ghostStreaming_) {
        dismissGhost();
    }
    if (pendingRangeActive_ || rangeStreaming_) {
        rejectPendingRange();
    }
}

void NoaCopilotModel::openInlineCard()
{
    if (!enabled_ || !document_.hasDocument() || document_.currentDifficultyId() <= 0) {
        return;
    }
    if (hasSkeleton(false)) { triggerManualContinue(); return; }
    dismissGhost();
    if (pendingRangeActive_) {
        rejectPendingRange();
    }
    inlineLevelOverride_ = document_.currentDifficultyLevel();
    inlineDesignerOverride_ = defaultDesigner_;
    inlineCardVisible_ = true;
    emit inlineCardChanged();
}

void NoaCopilotModel::closeInlineCard()
{
    if (!inlineCardVisible_) {
        return;
    }
    inlineCardVisible_ = false;
    emit inlineCardChanged();
}

void NoaCopilotModel::requestRangeGeneration(
    int spanBeats,
    bool fillToNext,
    const QString& levelOverride,
    const QString& designerOverride)
{
    const int requestedBeats = (spanBeats >= 0) ? spanBeats : inlineSpanBeats_;
    lastRangeFillToNext_ = fillToNext || (requestedBeats == 0) || inlineFillToNext_;
    lastRangeSpanBeats_ = requestedBeats > 0 ? requestedBeats : 8;
    lastRangeLevelOverride_ = levelOverride.isEmpty() ? inlineLevelOverride_ : levelOverride;
    lastRangeDesignerOverride_ = designerOverride.isEmpty() ? inlineDesignerOverride_ : designerOverride;
    sendSuggestRequest(
        QStringLiteral("range"),
        lastRangeSpanBeats_,
        lastRangeFillToNext_,
        lastRangeLevelOverride_,
        lastRangeDesignerOverride_);
}

void NoaCopilotModel::triggerRangeGenerate()
{
    requestRangeGeneration(
        inlineSpanBeats_,
        inlineFillToNext_ || (inlineSpanBeats_ == 0),
        inlineLevelOverride_,
        inlineDesignerOverride_);
}

bool NoaCopilotModel::acceptPendingRange()
{
    if (!pendingRangeActive_) {
        return false;
    }
    const QString text = pendingRangeText_;
    const int startPos = pendingReplaceStart_;
    const int endPos = pendingReplaceEnd_;
    pendingRangeActive_ = false;
    pendingRangeText_.clear();
    pendingReplaceStart_ = 0;
    pendingReplaceEnd_ = 0;
    emit pendingRangeChanged();
    updateEditorGhostStrike();
    preview_.setPlaybackRangeEnabled(false, 0.0, 0.0);
    clearDualSurfaceGhostPreview();
    return applyEditorReplacement(startPos, endPos, text);
}

void NoaCopilotModel::regeneratePendingRange()
{
    if (!pendingRangeActive_) {
        return;
    }
    sendSuggestRequest(
        QStringLiteral("range"),
        lastRangeSpanBeats_,
        lastRangeFillToNext_,
        lastRangeLevelOverride_,
        lastRangeDesignerOverride_);
}

void NoaCopilotModel::auditionPendingRange()
{
    if (!pendingRangeActive_) {
        return;
    }
    const double startSec = qMax(0.0, pendingStartSecond_ - 0.5);
    const double endSec = qMax(startSec + 1.0, pendingEndSecond_ + 0.5);
    preview_.setPlaybackRangeEnabled(true, startSec, endSec);
    preview_.setPositionSeconds(startSec);
    preview_.setPlaying(true);
}

void NoaCopilotModel::rejectPendingRange()
{
    pendingAcceptAfterStream_ = false;
    if (!pendingRangeActive_ && !rangeStreaming_) {
        return;
    }
    pendingRangeActive_ = false;
    rangeStreaming_ = false;
    pendingRangeText_.clear();
    pendingReplaceStart_ = 0;
    pendingReplaceEnd_ = 0;
    emit pendingRangeChanged();
    updateEditorGhostStrike();
    preview_.setPlaybackRangeEnabled(false, 0.0, 0.0);
    clearDualSurfaceGhostPreview();
}

void NoaCopilotModel::dismissEmptyCard()
{
    dismissedEmptyDifficultyId_ = document_.currentDifficultyId();
    emit emptyCardChanged();
}

void NoaCopilotModel::requestFullChartGeneration(
    const QString& levelOverride,
    const QString& designerOverride)
{
    if (!enabled_ || !document_.hasDocument() || document_.currentDifficultyId() <= 0) {
        return;
    }
    dismissGhost();
    if (pendingRangeActive_) {
        rejectPendingRange();
    }
    sendSuggestRequest(QStringLiteral("full"), 4, false, levelOverride, designerOverride);
}

void NoaCopilotModel::cancelActiveGeneration()
{
    if (activeRequestId_ <= 0) {
        return;
    }
    QJsonObject req;
    req.insert(QStringLiteral("op"), QStringLiteral("cancel"));
    req.insert(QStringLiteral("id"), activeRequestId_);
    writeDaemonJson(req);
    activeRequestId_ = 0;
    pendingAcceptAfterStream_ = false;
    if (ghostStreaming_) {
        ghostStreaming_ = false;
        dismissGhost();
    }
    if (rangeStreaming_) {
        rangeStreaming_ = false;
        rejectPendingRange();
    }
    endFullStream();
    discardCandidates();
    if (checkpointLoadedInDaemon_) {
        updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
    }
}

double NoaCopilotModel::currentChartBpm() const
{
    bool ok = false;
    const double whole = document_.wholeBpm().trimmed().toDouble(&ok);
    if (ok && whole > 0.0) {
        return whole;
    }
    static const QRegularExpression kBpmRe(QStringLiteral("\\((\\d+(?:\\.\\d+)?)\\)"));
    const QRegularExpressionMatch match = kBpmRe.match(document_.chartText());
    if (match.hasMatch()) {
        const double parsed = match.captured(1).toDouble(&ok);
        if (ok && parsed > 0.0) {
            return parsed;
        }
    }
    return 120.0;
}

double NoaCopilotModel::currentChartFirstSeconds() const
{
    return document_.metadataFirst().trimmed().toDouble();
}

double NoaCopilotModel::parseLevelNumeric(const QString& levelText) const
{
    QString cleaned = levelText.trimmed();
    const bool hasPlus = cleaned.endsWith(QLatin1Char('+'));
    if (hasPlus) {
        cleaned.chop(1);
    }
    bool ok = false;
    double val = cleaned.toDouble(&ok);
    if (!ok || val <= 0.0) {
        return 13.0;
    }
    if (hasPlus && qFloor(val) == val) {
        val += 0.7;
    }
    return qBound(1.0, val, 15.0);
}

QString NoaCopilotModel::effectiveDesigner(const QString& overrideDesigner) const
{
    if (!overrideDesigner.trimmed().isEmpty()) {
        return overrideDesigner.trimmed();
    }
    if (!defaultDesigner_.trimmed().isEmpty()) {
        return defaultDesigner_.trimmed();
    }
    return document_.currentDifficultyDesigner().trimmed();
}

} // namespace miacode::ui
