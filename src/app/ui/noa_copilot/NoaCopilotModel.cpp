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
#include "app/ui/noa_copilot/NoaCacheIndex.h"
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QMetaObject>
#include <QScopeGuard>
#include <QScopedValueRollback>
#include <QThread>
#include <memory>
#include <utility>
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
    if (bytes < 1024) return QStringLiteral("%1 B").arg(bytes);
    if (bytes < 1024 * 1024) return QStringLiteral("%1 KB").arg(double(bytes) / 1024.0, 0, 'f', 1);
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

} // namespace

void NoaCopilotModel::compileGhost(const QString& text, double start, double bpm, QVector<TimelineNoteMarker>& markers,
                  TimelineRenderSnapshot& timeline)
{
    QString chart = text.trimmed();
    if (chart.isEmpty()) { markers = {}; timeline = {}; return; }
    if (!chart.startsWith('(')) chart.prepend(QStringLiteral("(%1)").arg(bpm > 0 ? bpm : 160.0));
    if (!chart.endsWith('E', Qt::CaseInsensitive)) chart.append(QStringLiteral("\nE"));
    const auto parsed = SimaiParser::parseIncremental(chart, {}, ghostLineCache_);
    using namespace miacode::timeline::offset;
    markers = shiftedNoteMarkers(parsed.noteMarkers, start, NonFiniteHandling::PassThrough);
    if (!ghostTimelineModel_) ghostTimelineModel_ = std::make_unique<TimelineQuickModel>();
    ghostTimelineModel_->updateFromText(chart, start, {});
    timeline = ghostTimelineModel_->snapshot();
}

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
    exhaustionTimer_.setSingleShot(true);
    exhaustionTimer_.setInterval(2400);
    connect(&exhaustionTimer_, &QTimer::timeout, this, &NoaCopilotModel::ghostStateChanged);
    connect(&renderTimer_, &QTimer::timeout, this, &NoaCopilotModel::showCandidate);
    warmTimer_.setSingleShot(true);
    warmTimer_.setInterval(550);
    connect(&warmTimer_, &QTimer::timeout, this, &NoaCopilotModel::warmSong);
    connect(&document_, &DocumentModel::currentDifficultyFieldsChanged, this, [this] { warmTimer_.start(); });
    idleVramTimer_.setSingleShot(true);
    idleVramTimer_.setInterval(kIdleVramReleaseMs);
    connect(&idleVramTimer_, &QTimer::timeout, this, [this]() {
        if (device_ == QStringLiteral("cuda") && !activeRequestId_ && candidates_.isEmpty()) releaseVram();
    });

    // Ensure default checkpoints directory exists so users can drop .ckpt files into it.
    QDir().mkpath(defaultCheckpointsDir());

    rebuildDesignerOptions({});
    loadPreferences();
    refreshCheckpoints();
    if (QFileInfo::exists(QDir(defaultBuiltinRoot()).filePath(".miacode/.noacopilot")))
        noa_cache::rememberProject(defaultBuiltinRoot());
    scanEnvironments(false);

    connect(&document_, &DocumentModel::documentStateChanged, this, [this]() {
        if (!document_.hasDocument() || document_.wholeSourceEditorActive()) {
            dismissGhost();
        }
        sendBindSongIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::documentReplaced, this, [this]() {
        cancelActiveGeneration();
        discardCandidates();
        dismissGhost();
        dismissedEmptyDifficultyId_ = -1;
        lastObservedEditorText_ = document_.chartText();
        sendBindSongIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::currentFilePathChanged, this, [this]() {
        const auto chart = document_.currentFilePath();
        if (!chart.isEmpty() && QFileInfo::exists(QFileInfo(chart).dir().filePath(".miacode/.noacopilot")))
            noa_cache::rememberProject(QFileInfo(chart).absolutePath());
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
        dismissedEmptyDifficultyId_ = -1;
        lastObservedEditorText_ = document_.chartText();
        sendSyncChartIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::chartTextChanged, this, [this]() {
        // Only committed document changes invalidate a review. Scintilla also emits
        // text notifications when replacing the transient visual candidate.
        const QString chart = document_.chartText();
        if (chart != lastObservedEditorText_) {
            lastObservedEditorText_ = chart;
            if (!fullStreamingActive_ && !guidePracticeActive_) {
                notifyEditorCursorOrTextChanged(lastCursorPos_, lastSelStart_, lastSelEnd_, true);
            }
        }
        if (!fullStreamingActive_ && !guidePracticeActive_) warmTimer_.start();
        emit emptyCardChanged();
    });

    guideStreamTimer_.setInterval(75);
    connect(&guideStreamTimer_, &QTimer::timeout, this, [this]() {
        if (!guidePracticeActive_ || guideStage_ != 1) {
            guideStreamTimer_.stop();
            return;
        }
        if (guideStreamBeat_ < 16) {
            ++guideStreamBeat_;
            showGuideCandidate();
        } else {
            guideStreamTimer_.stop();
            guideStage_ = 2;
            showGuideCandidate();
            emit guidePracticeChanged();
        }
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
    return cpuPython();
}

QString NoaCopilotModel::cpuPython() const
{
    return cpuPython_.isEmpty() ? QDir(defaultBuiltinRoot()).filePath(QStringLiteral("env/python.exe")) : cpuPython_;
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
    return {
        QVariantMap{{QStringLiteral("value"), QStringLiteral("bundled")}, {QStringLiteral("label"), QStringLiteral("Miacode 内置推理源码")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("custom")}, {QStringLiteral("label"), QStringLiteral("自定义 Worktree / 推理源码目录")}},
    };
}

void NoaCopilotModel::loadPreferences()
{
    const QJsonObject root = PreferenceDocument::loadPreferencesObject();
    const QJsonObject sec = root.value(QLatin1String(kPrefSectionKey)).toObject();
    lineAlignBeats_ = qBound(1, sec["line_align_beats"].toInt(4), 64);
    continuationLines_ = qBound(1, sec["continuation_lines"].toInt(4), 64);
    candidateCount_ = qBound(1, sec["candidate_count"].toInt(3), 8);
    cpuThreads_ = sec["cpu_threads"].toInt(4);
    mertDevice_ = sec["mert_device"].toString(QStringLiteral("cpu"));
    cpuPython_ = sec["cpu_python"].toString();
    cudaPython_ = sec["cuda_python"].toString();
    enabled_ = sec.value(QStringLiteral("enabled")).toBool(true);
    defaultDesigner_ = sec.value(QStringLiteral("default_designer")).toString();
    device_ = sec.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    precision_ = sec.value(QStringLiteral("precision")).toString(QStringLiteral("bf16"));
    temperature_ = qBound(0.0, sec.value(QStringLiteral("temperature")).toDouble(0.5), 1.0);
    slideProbabilityMultiplier_ = qBound(1.0, sec.value(QStringLiteral("slide_probability_multiplier")).toDouble(1.0), 4.0);
    seed_ = sec.value(QStringLiteral("seed")).toInt(-1);
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
    if (!sec.contains("cpu_python") && pythonMode_ == "custom") cpuPython_ = customPythonPath_;
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
    sec["continuation_lines"] = continuationLines_;
    sec["candidate_count"] = candidateCount_;
    sec["cpu_threads"] = cpuThreads_;
    sec["mert_device"] = mertDevice_;
    sec["cpu_python"] = cpuPython_;
    sec["cuda_python"] = cudaPython_;
    sec.insert(QStringLiteral("enabled"), enabled_);
    sec.insert(QStringLiteral("default_designer"), defaultDesigner_);
    sec.insert(QStringLiteral("device"), device_);
    sec.insert(QStringLiteral("precision"), precision_);
    sec.insert(QStringLiteral("temperature"), temperature_);
    sec.insert(QStringLiteral("slide_probability_multiplier"), slideProbabilityMultiplier_);
    sec.insert(QStringLiteral("seed"), seed_);
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
        stopDaemon();
        updateStatus(QStringLiteral("disabled"), QStringLiteral("已关闭"));
    } else {
        ensureDaemonRunning();
    }
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
    warmTimer_.start();
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

void NoaCopilotModel::setContinuationLines(int value)
{
    value = qBound(1, value, 64);
    if (continuationLines_ == value) return;
    continuationLines_ = value;
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
    value = QDir::fromNativeSeparators(value.trimmed());
    if (value == QDir(defaultBuiltinRoot()).filePath(QStringLiteral("env/python.exe"))) value.clear();
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

void NoaCopilotModel::setSlideProbabilityMultiplier(double value)
{
    value = qBound(1.0, value, 4.0);
    if (qFuzzyCompare(slideProbabilityMultiplier_, value)) return;
    slideProbabilityMultiplier_ = value;
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

bool NoaCopilotModel::emptyCardVisible() const
{
    if (guidePracticeActive_ || !enabled_ || !document_.hasDocument() || document_.wholeSourceEditorActive()) {
        return false;
    }
    const int diffId = document_.currentDifficultyId();
    if (diffId <= 0 || diffId == dismissedEmptyDifficultyId_) {
        return false;
    }
    if (fullStreamingActive_ || ghostActive_) {
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

void NoaCopilotModel::browseEnvironmentPython(bool cpu)
{
    miacode::FileRequest req;
    req.title = cpu ? QStringLiteral("选择 CPU Python 解释器") : QStringLiteral("选择 CUDA Python 解释器");
    req.startPath = cpu ? cpuPython() : cudaPython_;
    req.nameFilters = {QStringLiteral("Python (python*.exe python python3*)"), QStringLiteral("All Files (*.*)")};
    uiRequests_.requestFile(req, [this, cpu](const QString& chosen) {
        if (chosen.isEmpty()) return;
        if (cpu) setCpuPython(chosen);
        else setCudaPython(chosen);
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
    if (cacheCleaning_) return;
    if (cacheScanActive_) cacheRefreshPending_ = true;
    else scanCacheIndex(false);
}

void NoaCopilotModel::scanCacheIndex(bool clear)
{
    cacheScanActive_ = true;
    cacheRefreshPending_ = false;
    auto result = std::make_shared<noa_cache::Usage>();
    auto* thread = QThread::create([clear, result] { *result = noa_cache::inspect(clear); });
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    connect(thread, &QThread::finished, this, [this, clear, result] {
        cacheScanActive_ = false;
        audioCacheSizeText_ = formatBytesText(result->bytes);
        if (clear) {
            cacheCleaning_ = false;
            cacheMessage_ = QStringLiteral("清理成功 %1 个，失败 %2 个").arg(result->removed).arg(result->failed);
            const auto queued = std::exchange(deferredCacheRequests_, {});
            for (const auto& request : queued) writeDaemonJson(request);
        } else if (cacheCleaning_ && !waitingForCacheBarrier_) {
            scanCacheIndex(true);
        } else if (cacheRefreshPending_ && !cacheCleaning_) {
            scanCacheIndex(false);
        }
        emit cacheInfoChanged();
    });
    thread->start();
}

void NoaCopilotModel::clearAudioCache()
{
    if (cacheCleaning_) return;
    cacheCleaning_ = true;
    cacheMessage_ = QStringLiteral("正在清理缓存…");
    emit cacheInfoChanged();
    if (supportsCacheBarrier_ && daemonConfigured_ && daemonProcess_ && daemonProcess_->state() == QProcess::Running) {
        waitingForCacheBarrier_ = true;
        writeDaemonJson({{"op","cache_barrier"},{"id",nextRpcId_++}});
    } else if (!cacheScanActive_) {
        scanCacheIndex(true);
    }
}

void NoaCopilotModel::finishCacheBarrier()
{
    if (!waitingForCacheBarrier_) return;
    waitingForCacheBarrier_ = false;
    if (!cacheScanActive_) scanCacheIndex(true);
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
            deferredCacheRequests_.clear();
            finishCacheBarrier();
            endFullStream();
            daemonConfigured_ = false;
            checkpointLoadedInDaemon_ = false;
            activeRequestId_ = 0;
            pendingBindSongId_ = 0;
            boundSongKey_.clear();
            lastWarmSignature_ = {};
            emit ghostStateChanged();
            if (enabled_) {
                updateStatus(
                    QStringLiteral("error"),
                    QStringLiteral("进程已退出 (%1)").arg(exitCode),
                    statusDetail_);
            }
        });

    updateStatus(QStringLiteral("starting"), QStringLiteral("正在启动..."));
    const QStringList args{QStringLiteral("-B"), QStringLiteral("-u"), QStringLiteral("-m"), QStringLiteral("maidata.spine.copilot")};
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
    deferredCacheRequests_.clear();
    finishCacheBarrier();
    endFullStream();
    stdoutBuffer_.clear();
    warmTimer_.stop();
    idleVramTimer_.stop();
    activeRequestId_ = 0;
    emit ghostStateChanged();
    pendingBindSongId_ = 0;
    daemonConfigured_ = false;
    checkpointLoadedInDaemon_ = false;
    boundSongKey_.clear();
    lastWarmSignature_ = {};

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
    const auto op = payload["op"].toString();
    if (cacheCleaning_ && op != "cache_barrier" && op != "close") {
        deferredCacheRequests_.append(payload);
        return;
    }
    const QString audio = payload["audio"].toString();
    if (!audio.isEmpty() && (op == "bind_song" || op == "suggest" || op == "warm"))
        noa_cache::rememberProject(QFileInfo(audio).absolutePath());
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
    // Older custom daemons require this configure field even before a song is open.
    const auto song = document_.currentFilePath();
    const QDir base(song.isEmpty() ? defaultBuiltinRoot() : QFileInfo(song).absolutePath());
    noa_cache::rememberProject(base.absolutePath());
    req.insert(QStringLiteral("cache_dir"), base.filePath(".miacode/.noacopilot"));
    writeDaemonJson(req);
}

void NoaCopilotModel::sendBindSongIfNeeded()
{
    if (guidePracticeActive_ || !enabled_ || !document_.hasDocument()) {
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
    lastWarmSignature_ = {};
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
    if (needsChartSync_) req.insert(QStringLiteral("text"), document_.chartText());
    writeDaemonJson(req);
}

void NoaCopilotModel::sendSyncChartIfNeeded()
{
    if (guidePracticeActive_ || !needsChartSync_ || !enabled_ || !daemonConfigured_ || !document_.hasDocument() || document_.currentDifficultyId() <= 0) {
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
    if (mode == QStringLiteral("continue")) req.insert(QStringLiteral("continuation_lines"), continuationLines_);
    req.insert(QStringLiteral("temperature"), temperature_);
    req.insert(QStringLiteral("slide_probability_multiplier"), slideProbabilityMultiplier_);
    req.insert(QStringLiteral("seed"), seed_);
    if (mode == "continue") {
        if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge()))
            continuationSecond_ = bridge->cursorSeconds();
    }
    request_ = req;
    prefetchThrough_ = mode == "full" ? 1 : candidateCount_;
    emit ghostStateChanged();
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
        {"first",currentChartFirstSeconds()},{"bpm",currentChartBpm()},
        {"level",parseLevelNumeric(document_.currentDifficultyLevel())},{"difficulty",document_.currentDifficultyId()},
        {"designer",effectiveDesigner()},{"line_align_beats",lineAlignBeats_},{"candidate_count",candidateCount_}};
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
    const QString cpu = cpuPython();
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
        cpuAvailable_ = cudaAvailable_ = cpuMertAvailable_ = cudaMertAvailable_ = false;
        QString reason;
        for (const auto& v : *rows) {
            const auto row=v.toMap();
            const QString path=row["path"].toString();
            if (path==QDir::fromNativeSeparators(cpu)) {
                cpuAvailable_ = row["cpu"].toBool();
                cpuMertAvailable_ = cpuAvailable_ && row["mert"].toBool();
                if (cpuAvailable_ && !cpuMertAvailable_) reason = row["mertReason"].toString();
                if (!cpuAvailable_) reason = row["reason"].toString();
            }
            if (path==QDir::fromNativeSeparators(cuda)) {
                cudaAvailable_=row["cuda"].toBool();
                cudaMertAvailable_ = cudaAvailable_ && row["mert"].toBool();
                if (cudaAvailable_ && !cudaMertAvailable_) reason = row["mertReason"].toString();
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
    req["text"] = noaWarmTimingText(document_.chartText());
    req["valid_bpm"] = document_.wholeBpm().toDouble() > 0 || req["text"].toString().contains('(');
    req.remove("candidate_count");
    req.remove("line_align_beats");
    if (req == lastWarmSignature_) return;
    lastWarmSignature_ = req;
    req["op"] = "warm";
    req["id"] = nextRpcId_++;
    writeDaemonJson(req);
}

bool NoaCopilotModel::selectionHasSkeleton() const
{
    if (document_.wholeSourceEditorActive() || !document_.hasDocument()) return false;
    const QString body = document_.chartText();
    const int lo = qMin(lastSelStart_,lastSelEnd_), hi = qMax(lastSelStart_,lastSelEnd_);
    if (lo == hi) return false;
    const int a = lo > 0 ? int(body.lastIndexOf('\n', lo - 1)) + 1 : 0;
    const int nl = body.indexOf('\n', hi);
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
        if (a + match.capturedStart(1) < hi && a + match.capturedEnd(1) > lo) return true;
    }
    return false;
}

void NoaCopilotModel::mouseSkeletonSelection(bool quick)
{
    skeletonSelection_ = enabled_ && selectionHasSkeleton();
    emit ghostStateChanged();
    if (skeletonSelection_ && quick) triggerManualContinue();
}

void NoaCopilotModel::discardCandidates()
{
    renderTimer_.stop();
    exhaustionTimer_.stop();
    strikeLeft_ = strikeRight_ = -1;
    ghostLineCache_.reset();
    ghostTimelineModel_.reset();
    if (!candidates_.isEmpty()) {
        preview_.setCopilotReview(false);
        if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) bridge->clearGhostRangeCache();
        if (activeRequestId_) writeDaemonJson({{"op","cancel"},{"id",activeRequestId_}});
        writeDaemonJson({{"op","discard"},{"id",nextRpcId_++}});
        activeRequestId_ = 0;
    }
    selected_ = highCandidate_ = prefetchThrough_ = 0;
    exhausted_ = false;
    candidates_.clear();
    stepRequest_ = {};
}

void NoaCopilotModel::requestMore()
{
    if (activeRequestId_ || exhausted_ || highCandidate_ >= prefetchThrough_) return;
    activeRequestId_ = nextRpcId_++;
    QJsonObject req = orderedSteps_ ? QJsonObject{{"protocol", 2}, {"anchor", request_["id"]},
                                                 {"revision", qint64(activeRequestRevision_)}} : request_;
    req["op"] = "more";
    req["target"] = prefetchThrough_;
    req["id"] = activeRequestId_;
    writeDaemonJson(req);
}

void NoaCopilotModel::selectBackendCandidate()
{
    writeDaemonJson({{"op", "select"}, {"id", nextRpcId_++}, {"anchor", request_["id"]}, {"candidate", selected_}});
}

void NoaCopilotModel::notifyHint(const QString& message)
{
    hintText_ = message;
    exhaustionTimer_.start();
    emit ghostStateChanged();
}

void NoaCopilotModel::beginCandidate(int number, const QJsonObject& layout)
{
    Candidate c;
    c.writer.begin(layout, document_.chartText(), request_["line_align_beats"].toInt(4));
    c.left = c.cut = layout["left"].toInt();
    c.right = c.origEnd = layout["right"].toInt();
    continuationEndSecond_ = layout["end_second"].toDouble();
    c.startSecond = layout["start_second"].toDouble();
    c.activeBpm = c.writer.bpm(layout["start"].toInt());
    candidates_.insert(number, std::move(c));
}

void NoaCopilotModel::switchCandidate(int direction)
{
    if (!candidateSessionActive()) return;
    const auto candidate = candidates_.constFind(selected_);
    const double anchor = candidate == candidates_.cend() ? continuationSecond_ : candidate->startSecond;
    const auto seekAfterSwitch = qScopeGuard([this, anchor] { preview_.setPositionSeconds(anchor); });
    int next = selected_ + direction;
    if (direction < 0) {
        while (next > 0 && !candidates_.contains(next)) --next;
        if (next <= 0) {
            dismissAllActivePreviews();
            emit returnedToOriginal();
            return;
        }
    } else {
        const auto current = candidates_.constFind(selected_);
        if (current == candidates_.cend() || !current->ready) {
            if (current != candidates_.cend() || !ghostActive_)
                notifyHint(QStringLiteral("诺亚还没写完哦，请稍等"));
            return;
        }
        while (next <= highCandidate_ && !candidates_.contains(next)) ++next;
        if (exhausted_ && next > highCandidate_) {
            notifyHint(QStringLiteral("诺亚想不出来更多的写法了哦"));
            return;
        }
        prefetchThrough_ = qMax(prefetchThrough_, next + candidateCount_ - 1);
        if (!candidates_.contains(next)) {
            QJsonObject layout = current->writer.layout();
            if (current->stepsTaken) {
                layout["start"] = request_["step_tick"];
                layout["division"] = request_["division"];
                layout["lead"] = "";
                // The retained writer's slot coordinates precede the committed beat.
                // Hold this provisional range silent until its new descriptor arrives.
                layout["skeleton"] = false;
                layout["left"] = current->left;
                layout["right"] = current->right;
                layout["start_second"] = current->startSecond;
            }
            beginCandidate(next, layout);
        }
    }
    pendingAcceptAfterStream_ = false;
    exhaustionTimer_.stop();
    renderTimer_.stop();
    selected_ = next;
    selectBackendCandidate();
    showCandidate();
    if (direction > 0) requestMore();
}

void NoaCopilotModel::prepareCandidatePreview(Candidate& candidate)
{
    const QString replacement = candidate.stepsTaken ? candidate.remainder : candidate.writer.auditionText();
    if (candidate.previewReady && candidate.previewKey == replacement && (!candidate.ready || candidate.analysisId)) return;
    const QString source = document_.chartText();
    const QString chart = source.left(candidate.left) + replacement + source.mid(candidate.right);
    const auto& layout = candidate.writer.layout();
    const int fixedLead = candidate.stepsTaken ? 0 : layout["lead"].toString().size();
    const NoaAuditionRange range{utf16OffsetToLineCol(chart, candidate.left + fixedLead),
                                 utf16OffsetToLineCol(chart, candidate.left + replacement.size())};
    const auto timing = document_.timingMetadata();
    const double first = currentChartFirstSeconds();
    candidate.audition = noaAuditionMarkers(chart, range, first, timing, ghostLineCache_);
    if (!ghostTimelineModel_) ghostTimelineModel_ = std::make_unique<TimelineQuickModel>();
    ghostTimelineModel_->updateFromText(chart, first, timing);
    candidate.normalTimeline = candidate.timeline = ghostTimelineModel_->snapshot();
    for (qsizetype i = 0; i < candidate.timeline.lines.size(); ++i) {
        const auto& line = candidate.normalTimeline.lines.at(i);
        if (line.lineNumber < range.first.first || line.lineNumber > range.end.first) {
            candidate.timeline.lines[i].notes.clear();
        } else if (line.lineNumber > range.first.first && line.lineNumber < range.end.first) {
            candidate.normalTimeline.lines[i].notes.clear();
        } else {
            const int lineNumber = line.lineNumber;
            auto generated = [&](const auto& note) { return range.contains(lineNumber, note.sourceCol); };
            candidate.normalTimeline.lines[i].notes.removeIf(generated);
            candidate.timeline.lines[i].notes.removeIf([&](const auto& note) { return !generated(note); });
        }
    }
    candidate.previewReady = true;
    candidate.previewKey = replacement;
    if (!candidate.ready) return;
    const quint64 id = candidate.analysisId = ++nextAnalysisId_;
    const quint64 revision = document_.documentRevision();
    candidate.localIssues = 0;
    preview_.localGhostIssues(candidate.audition.ghost, candidate.startSecond, candidate.writer.layout()["end_second"].toDouble())
        .then(this, [this, id, revision](int count) {
            if (document_.documentRevision() != revision) return;
            for (auto it = candidates_.begin(); it != candidates_.end(); ++it) {
                if (it->analysisId != id) continue;
                it->localIssues = count;
                if (it.key() == selected_) emit ghostStateChanged();
                return;
            }
        });
}

bool NoaCopilotModel::seekCandidateAt(int visualPosition)
{
    const auto it = candidates_.constFind(selected_);
    if (it == candidates_.cend() || visualPosition < it->left
        || visualPosition >= it->left + ghostText_.size() || !it->previewReady) return false;
    const auto& lines = it->timeline.lines;
    const auto end = std::upper_bound(lines.cbegin(), lines.cend(), visualPosition,
        [](int position, const auto& line) { return position < line.startPosition; });
    if (end == lines.cbegin()) return false;
    const auto& line = *std::prev(end);
    const int col = visualPosition - line.startPosition + 1;
    const auto note = std::upper_bound(line.notes.cbegin(), line.notes.cend(), col,
        [](int column, const auto& entry) { return column < entry.sourceCol; });
    const auto beat = std::upper_bound(line.beats.cbegin(), line.beats.cend(), col,
        [](int column, const auto& entry) { return column < entry.sourceCol; });
    const bool hasNote = note != line.notes.cbegin();
    const bool hasBeat = beat != line.beats.cbegin();
    if (!hasNote && !hasBeat) return false;
    const double offset = hasNote && (!hasBeat || std::prev(note)->sourceCol >= std::prev(beat)->sourceCol)
        ? std::prev(note)->secondOffset : std::prev(beat)->secondOffset;
    preview_.setPositionSeconds(line.startSecond + offset);
    return true;
}

void NoaCopilotModel::showCandidate()
{
    const auto it = candidates_.find(selected_);
    if (it == candidates_.end()) return;
    auto& c = it.value();
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
    updateGhostSplitLines();
    emit ghostStateChanged();
    updateEditorGhostStrike();
    prepareCandidatePreview(c);
    preview_.setGhostPreviewOverlay(c.audition.ghost, false, 0, 0, 0, 0, !c.ready, &c.audition);
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge()))
        bridge->setCandidateSnapshots(c.normalTimeline, c.timeline);
}

bool NoaCopilotModel::handleStructured(const QJsonObject& m)
{
    const QString event=m["event"].toString();
    if (event!="candidate_begin" && event!="moment" && event!="beat" && event!="candidate_end" && event!="pool_end" && event!="candidate_reset")
        return false;
    if (m["id"].toInt()!=activeRequestId_ || activeRequestRevision_!=document_.documentRevision()) return true;
    const int number=m["candidate"].toInt();
    if (event=="candidate_begin") {
        preview_.setCopilotReview(true);
        const bool first = candidates_.isEmpty() && number == 1;
        beginCandidate(number, m);
        if (first) selected_ = 1;
        if (number == selected_ && !fullStreamingActive_) {
            renderTimer_.stop();
            showCandidate();
        }
        return true;
    }
    if (event == "candidate_reset") {
        if (candidates_.contains(number)) {
            const auto layout = candidates_[number].writer.layout();
            beginCandidate(number, layout);
            if (selected_ == number) { renderTimer_.stop(); showCandidate(); }
        }
        return true;
    }
    if (event == "pool_end") {
        activeRequestId_ = 0;
        exhausted_ = m["exhausted"].toBool();
        if (exhausted_) {
            const bool requestedMissing = selected_ > highCandidate_ || highCandidate_ == 0;
            if (highCandidate_) candidates_.remove(highCandidate_ + 1);
            if (!highCandidate_) dismissGhost();
            else if (requestedMissing) {
                pendingAcceptAfterStream_ = false;
                selected_ = highCandidate_;
                selectBackendCandidate();
                showCandidate();
            }
            if (requestedMissing) notifyHint(QStringLiteral("诺亚想不出来更多的写法了哦"));
        } else {
            requestMore();
        }
        if (!activeRequestId_) updateStatus("ready", QStringLiteral("就绪"), activeCheckpointName());
        emit ghostStateChanged();
        return true;
    }
    if (!candidates_.contains(number)) return true;
    auto& c=candidates_[number];
    if (event=="candidate_end") {
        c.writer.finish();
        c.ready=true;
        if (!fullStreamingActive_) prepareCandidatePreview(c);
        highCandidate_=qMax(number,highCandidate_);
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
        while (candidates_.size() > 16) {
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
            if (!renderTimer_.isActive()) renderTimer_.start(fullStreamingActive_ ? 200 : 65);
        }
    }
    return true;
}

void NoaCopilotModel::handleDaemonMessage(const QJsonObject& message)
{
    if (handleStructured(message)) return;
    const bool hadSession = candidateSessionActive();
    const auto notifySession = qScopeGuard([&] {
        if (hadSession != candidateSessionActive()) emit ghostStateChanged();
    });
    const QString event = message.value(QStringLiteral("event")).toString();
    if ((event == "result" || event == "stream_chunk") && message["id"].toInt() == activeRequestId_
        && message.contains("end_second")) continuationEndSecond_ = message["end_second"].toDouble();
    if (event == "cache_barrier") { finishCacheBarrier(); return; }
    if (event == "cache_updated") { if (cacheInfoVisible_) refreshAudioCacheSize(); return; }
    if (event == QStringLiteral("configured")) {
        supportsCacheBarrier_ = message["cache_barrier"].toBool();
        daemonConfigured_ = message.value(QStringLiteral("ok")).toBool();
        checkpointLoadedInDaemon_ = message.value(QStringLiteral("checkpoint_loaded")).toBool();
        orderedSteps_ = message.value(QStringLiteral("ordered_steps")).toBool();
        needsChartSync_ = message.value(QStringLiteral("sync_chart")).toBool(true);
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
        if (cacheInfoVisible_) refreshAudioCacheSize();
        if (activeRequestId_ > 0) {
            if (statusState_ == QStringLiteral("warming_audio")) {
                const QString genLabel = activeRequestMode_ == QStringLiteral("full")
                    ? QStringLiteral("正在流式生成整曲...")
                    : QStringLiteral("正在续写...");
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
        const double activeBpm = message.value(QStringLiteral("active_bpm")).toDouble(currentChartBpm());
        const QString ghostSimai = message.value(QStringLiteral("ghost_simai")).toString();

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
            endFullStream();
            updateStatus(QStringLiteral("error"), QStringLiteral("生成失败"), err);
            return;
        }

        updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
        if (cacheInfoVisible_) refreshAudioCacheSize();

        if (activeRequestRevision_ != document_.documentRevision()) {
            activeRequestId_ = 0;
            pendingAcceptAfterStream_ = false;
            if (ghostStreaming_) {
                ghostStreaming_ = false;
                dismissGhost();
            }
            endFullStream();
            return;
        }
        const QString text = message.value(QStringLiteral("text")).toString();
        const int replaceStart = message.value(QStringLiteral("replace_start")).toInt(lastCursorPos_);
        const int replaceEnd = message.value(QStringLiteral("replace_end")).toInt(lastCursorPos_);
        const double startSec = message.value(QStringLiteral("start_second")).toDouble(currentChartFirstSeconds());
        const double activeBpm = message.value(QStringLiteral("active_bpm")).toDouble(currentChartBpm());
        const QString ghostSimai = message.value(QStringLiteral("ghost_simai")).toString();

        if (mode == QStringLiteral("full")) {
            if (publishFullChart(text)) streamingProgress_ = 1.0;
            activeRequestId_ = 0;
            pendingAcceptAfterStream_ = false;
            endFullStream();
            return;
        }

        activeRequestId_ = 0;
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
    continuationSecond_ = startSecond;
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
    compileGhost(ghostSimai, startSecond, activeBpm, ghostMarkers, ghostTimelineSnapshot);

    // 1. Circular Preview + Playback SFX (excludes struck-through notes & gives ghost notes identical SFX)
    preview_.setGhostPreviewOverlay(
        ghostMarkers,
        hasStrike,
        strikeStartLine,
        strikeStartCol,
        strikeEndLine,
        strikeEndCol);

    // 2. Bottom Timeline Ghost Notes + Struck-through exclusion
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        bridge->setGhostStrikeRange(hasStrike, strikeStartLine, strikeStartCol, strikeEndLine, strikeEndCol);
        bridge->setGhostSnapshot(ghostTimelineSnapshot);
    }
}

void NoaCopilotModel::clearDualSurfaceGhostPreview()
{
    preview_.clearGhostPreviewOverlay(acceptingStep_);
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        bridge->clearCandidateSnapshots(acceptingStep_);
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
    if (guidePracticeActive_) {
        lastCursorPos_ = cursorPos;
        lastSelStart_ = selStart;
        lastSelEnd_ = selEnd;
        return;
    }
    lastCursorPos_ = cursorPos;
    lastSelStart_ = selStart;
    lastSelEnd_ = selEnd;
    idleVramTimer_.start();

    if (textChanged && activeRequestId_ > 0
        && activeRequestMode_ == QStringLiteral("continue")) {
        cancelActiveGeneration();
    }
    if (ghostActive_ && textChanged) {
        dismissGhost();
    }
    const bool selection = selectionHasSkeleton();
    if (selection != skeletonSelection_) { skeletonSelection_ = selection; emit ghostStateChanged(); }
    if (textChanged && !fullStreamingActive_) {
        sendSyncChartIfNeeded();
    }
}

bool NoaCopilotModel::candidateSessionActive() const
{
    if (guidePracticeActive_) return guideStage_ >= 1 && guideStage_ <= 7;
    return enabled_ && !fullStreamingActive_ && (ghostActive_ || !candidates_.isEmpty()
        || (activeRequestId_ > 0 && activeRequestMode_ == QStringLiteral("continue")));
}

bool NoaCopilotModel::handleCandidateKey(int key)
{
    if (guidePracticeActive_) return handleGuideKey(key);
    if (!candidateSessionActive()) return false;
    if (key == Qt::Key_Up || key == Qt::Key_Down) {
        switchCandidate(key == Qt::Key_Up ? -1 : 1);
        return true;
    }
    if (key != Qt::Key_Tab) return false;
    if (!ghostActive_ || ghostStreaming_) pendingAcceptAfterStream_ = true;
    else acceptFullGhost();
    return true;
}

bool NoaCopilotModel::handleEditorKeyPress(int key, int modifiers)
{
    if (guidePracticeActive_) {
        if (modifiers == Qt::NoModifier) {
            return handleGuideKey(key);
        }
        return false;
    }
    if (!enabled_) {
        return false;
    }
    const bool ctrl = (modifiers & Qt::ControlModifier) != 0;
    const bool shift = (modifiers & Qt::ShiftModifier) != 0;
    const bool alt = (modifiers & Qt::AltModifier) != 0;
    const bool meta = (modifiers & Qt::MetaModifier) != 0;
    const bool plainTab = (key == Qt::Key_Tab && !ctrl && !shift && !alt && !meta);

    if (modifiers == Qt::NoModifier && handleCandidateKey(key)) return true;
    if (!candidates_.isEmpty() && !fullStreamingActive_ && modifiers == Qt::NoModifier) {
        if (key == Qt::Key_Right) {
            if (selected_ && !ghostStreaming_) acceptNextGhostStep();
            return true;
        }
        if (key == Qt::Key_Escape) { dismissGhost(); return true; }
    }
    if (ghostActive_) {
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

    if (key == Qt::Key_Escape && (fullStreamingActive_ || activeRequestId_ > 0)) {
        cancelActiveGeneration();
        return true;
    }

    // When NoaCopilot is enabled, plain Tab triggers continuation instead of
    // Scintilla's default tab/indent block fill; disabling NoaCopilot restores Tab.
    if (plainTab && !fullStreamingActive_
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
    sendSuggestRequest(QStringLiteral("continue"), QString(), QString());
}

bool NoaCopilotModel::acceptFullGhost()
{
    if (ghostStreaming_) { pendingAcceptAfterStream_ = true; return true; }
    if (!ghostActive_) return false;
    if (ghostText_.isEmpty() && ghostReplaceStart_ == ghostReplaceEnd_) { dismissGhost(); return true; }
    const QString textToInsert = ghostText_;
    const int startPos = ghostReplaceStart_;
    const int endPos = ghostReplaceEnd_;
    QScopedValueRollback guard(acceptingStep_, true);
    if (!applyEditorReplacement(startPos, endPos, textToInsert)) return false;
    dismissGhost();
    return true;
}

bool NoaCopilotModel::acceptNextGhostStep()
{
    if (!ghostActive_ || ghostStreaming_ || ghostText_.isEmpty()) {
        return false;
    }
    auto acceptToEnd = [this] {
        const double end = continuationEndSecond_;
        if (!acceptFullGhost()) return false;
        preview_.setPositionSeconds(end);
        return true;
    };
    if (!candidates_.isEmpty() && selected_ && candidates_[selected_].ready) {
        const auto& writer = candidates_[selected_].writer;
        const auto step = writer.step(ghostCurrentStep_);
        if (step.isEmpty()) return acceptToEnd();
        ghostSteps_ = QJsonArray{step,QJsonObject()};
        ghostCurrentStep_ = 0;
        stepRequest_ = request_;
        stepRequest_["step_tick"] = step["tick"];
        stepRequest_["candidate"] = selected_;
        stepRequest_["division"] = step["division"];
    }
    if (ghostSteps_.isEmpty() || ghostCurrentStep_ >= ghostSteps_.size()) {
        return acceptToEnd();
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
        return acceptToEnd();
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
    if (!applyEditorReplacement(insertStart, consumeEnd, replacement, nextCursor)) {
        ghostActive_ = true;
        return false;
    }

    const auto seekAfterStep = qScopeGuard([this, remStartSec] { preview_.setPositionSeconds(remStartSec); });
    continuationSecond_ = remStartSec;
    if (!hasMoreSteps) {
        dismissGhost();
        return true;
    }

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
        selected_ = highCandidate_ = 1; prefetchThrough_ = candidateCount_; exhausted_ = false;
        const int anchorId = nextRpcId_++;
        activeRequestRevision_ = document_.documentRevision();
        lastCursorPos_ = lastSelStart_ = lastSelEnd_ = nextCursor;
        lastObservedEditorText_ = document_.chartText();
        stepRequest_["accepted_steps"] = prior+1;
        stepRequest_["anchor"] = request_["id"];
        stepRequest_["id"] = anchorId;
        if (orderedSteps_) stepRequest_["op"] = "step_anchor";
        stepRequest_["revision"] = qint64(activeRequestRevision_);
        stepRequest_["text"] = lastObservedEditorText_;
        stepRequest_["cursor_offset"] = nextCursor;
        stepRequest_["sel_start"] = stepRequest_["sel_end"] = nextCursor;
        stepRequest_["replace_start"] = nextCursor;
        stepRequest_["replace_end"] = ghostReplaceEnd_;
        request_ = stepRequest_;
        ghostCurrentStep_ = prior+1;
        writeDaemonJson(stepRequest_);
        // The FIFO worker commits each anchor even when a later key cancels its refill.
        activeRequestId_ = anchorId;
        if (orderedSteps_) {
            activeRequestId_ = nextRpcId_++;
            writeDaemonJson({{"op", "refill"}, {"protocol", 2}, {"id", activeRequestId_},
                             {"anchor", anchorId}, {"revision", qint64(activeRequestRevision_)}});
        }
        stepRequest_ = {};
        showCandidate();
    } else {
        applyDualSurfaceGhostPreview(remGhostSimai, remStartSec, remBpm, ghostReplaceStart_, ghostReplaceEnd_);
    }
    return true;
}

void NoaCopilotModel::dismissGhost()
{
    discardCandidates();
    pendingAcceptAfterStream_ = false;
    if (!ghostActive_ && !ghostStreaming_ && ghostText_.isEmpty() && ghostReplaceStart_ == 0 && ghostReplaceEnd_ == 0) {
        emit ghostStateChanged();
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
    if (!fullStreamingActive_) {
        clearDualSurfaceGhostPreview();
    }
}

void NoaCopilotModel::dismissAllActivePreviews()
{
    if (guidePracticeActive_) {
        if (guideStage_ >= 1 && guideStage_ <= 7) {
            QTimer::singleShot(0, this, [this]() {
                if (guidePracticeActive_ && guideStage_ >= 1 && guideStage_ <= 7) {
                    showGuideCandidate();
                }
            });
        }
        return;
    }
    if (activeRequestId_ > 0
        && activeRequestMode_ == QStringLiteral("continue")) {
        cancelActiveGeneration();
    }
    if (ghostActive_ || ghostStreaming_) {
        dismissGhost();
    }
}

namespace {
const QStringList& guideGroup1Beats()
{
    static const QStringList beats = {
        QStringLiteral("{4}4/7>8-5p4[4:3],"), QStringLiteral(","), QStringLiteral(","), QStringLiteral(","),
        QStringLiteral("\n{4},"), QStringLiteral(","), QStringLiteral(","), QStringLiteral(","),
        QStringLiteral("\n{4},"), QStringLiteral(","), QStringLiteral(","), QStringLiteral(","),
        QStringLiteral("\n{4},"), QStringLiteral(","), QStringLiteral(","), QStringLiteral(","),
    };
    return beats;
}

QString guideGroup2Full()
{
    return QStringLiteral(
        "{12}24,3,3,64,5,5,38,2,1,68,7,8,\n"
        "{32}1b/4b,,,,,,2b/7b,,,,,,8b/5b,,,,3,4,3,4,3,4,3,4,25,,,,,,,,\n"
        "{12}7,5,6,3,5,4,6,5,5,4,5,5,\n"
        "{6}6b-8[16:1],2b-4[16:1],1b-5[16:1],3b-8[16:1],7b-4[16:1],1b,");
}

QString guideGroup3Full()
{
    return QStringLiteral(
        "{32}4,6,4,6,4,6,4,6,3,7,3,7,2,8,2,8,1,1,1,1,1,,,,B1/B2/E2f/A2/D2/E3,,,,,,,,\n"
        "{32}5,3,5,3,5,3,5,3,6,2,6,2,7,1,7,1,8,8,8,8,8,,,,B7/E7/A7/B8/E8f/D8,,,,,,,,\n"
        "{32}1,2,3,4,5,6,7,8,1,2,3,4,1,2,3,4,8,7,6,5,4,3,2,1,8,7,6,5,8,7,6,5,\n"
        "{32}2,3,4,5,2,3,4,5,6pp4[4:1]*pp4[4:1],7,8,1,2,3,4,5,,,,,,,,,,,,,,,,,");
}
} // namespace

int NoaCopilotModel::guidePracticeCompletedSteps() const
{
    if (guideStage_ <= 0) return 0;
    if (guideStage_ == 1 || guideStage_ == 2) return 1;
    if (guideStage_ == 3) return 2;
    if (guideStage_ == 4) return 3;
    if (guideStage_ == 5) return 4;
    if (guideStage_ == 6) return 5;
    if (guideStage_ == 7) return 6;
    return 7;
}

QString NoaCopilotModel::guidePracticeExpectedKey() const
{
    if (guideStage_ == 0 || guideStage_ == 7) return QStringLiteral("Tab");
    if (guideStage_ == 2 || guideStage_ == 3) return QStringLiteral("↓");
    if (guideStage_ == 4) return QStringLiteral("↑");
    if (guideStage_ == 5 || guideStage_ == 6) return QStringLiteral("→");
    return {};
}

QString NoaCopilotModel::guidePracticePrompt() const
{
    switch (guideStage_) {
    case 0: return qtTrId("guide.noa.sandbox.prompt.tab_start");
    case 1: return qtTrId("guide.noa.sandbox.prompt.streaming");
    case 2: return qtTrId("guide.noa.sandbox.prompt.down_1");
    case 3: return qtTrId("guide.noa.sandbox.prompt.down_2");
    case 4: return qtTrId("guide.noa.sandbox.prompt.up");
    case 5: return qtTrId("guide.noa.sandbox.prompt.right_1");
    case 6: return qtTrId("guide.noa.sandbox.prompt.right_2");
    case 7: return qtTrId("guide.noa.sandbox.prompt.tab_accept");
    default: return qtTrId("guide.noa.sandbox.prompt.done");
    }
}

QString NoaCopilotModel::guideCandidateText() const
{
    if (guideCandidate_ == 1) {
        const auto& beats = guideGroup1Beats();
        const int count = qBound(0, guideStreamBeat_, int(beats.size()));
        QString joined;
        for (int i = 0; i < count; ++i) joined += beats.at(i);
        return guidePrefixLead_ + joined;
    }
    if (guideCandidate_ == 2) {
        const QString full = guideGroup2Full();
        if (guideAcceptedBeats_ == 1) {
            return full.mid(QStringLiteral("{12}24,3,3,").size());
        }
        if (guideAcceptedBeats_ >= 2) {
            return full.mid(QStringLiteral("{12}24,3,3,64,5,5,").size());
        }
        return guidePrefixLead_ + full;
    }
    if (guideCandidate_ == 3) {
        return guidePrefixLead_ + guideGroup3Full();
    }
    return {};
}

void NoaCopilotModel::showGuideCandidate()
{
    if (!guidePracticeActive_) return;
    const QString text = guideCandidateText();
    const QString source = document_.chartText();
    const int insertPos = qBound(0, guideInsertPos_, int(source.size()));
    guideInsertPos_ = insertPos;

    ghostActive_ = !text.isEmpty();
    ghostStreaming_ = (guideStage_ == 1);
    ghostText_ = text;
    ghostReplaceStart_ = insertPos;
    ghostReplaceEnd_ = insertPos;
    selected_ = guideCandidate_;
    updateGhostSplitLines();
    emit ghostStateChanged();
    updateEditorGhostStrike();

    if (text.isEmpty()) return;

    const QString chart = source.left(insertPos) + text + source.mid(insertPos);
    const int leadLen = (guideAcceptedBeats_ == 0) ? guidePrefixLead_.size() : 0;
    const NoaAuditionRange range{
        utf16OffsetToLineCol(chart, insertPos + leadLen),
        utf16OffsetToLineCol(chart, insertPos + text.size())
    };
    const auto timing = document_.timingMetadata();
    const double first = currentChartFirstSeconds();
    const auto audition = noaAuditionMarkers(chart, range, first, timing, ghostLineCache_);
    preview_.setCopilotReview(true);
    preview_.setGhostPreviewOverlay(audition.ghost, false, 0, 0, 0, 0, guideStage_ == 1, &audition);

    if (!ghostTimelineModel_) ghostTimelineModel_ = std::make_unique<TimelineQuickModel>();
    ghostTimelineModel_->updateFromText(chart, first, timing);
    TimelineRenderSnapshot normalTimeline = ghostTimelineModel_->snapshot();
    TimelineRenderSnapshot ghostTimeline = normalTimeline;
    for (qsizetype i = 0; i < ghostTimeline.lines.size(); ++i) {
        const auto& line = normalTimeline.lines.at(i);
        if (line.lineNumber < range.first.first || line.lineNumber > range.end.first) {
            ghostTimeline.lines[i].notes.clear();
        } else if (line.lineNumber > range.first.first && line.lineNumber < range.end.first) {
            normalTimeline.lines[i].notes.clear();
        } else {
            const int lineNumber = line.lineNumber;
            auto generated = [&](const auto& note) { return range.contains(lineNumber, note.sourceCol); };
            normalTimeline.lines[i].notes.removeIf(generated);
            ghostTimeline.lines[i].notes.removeIf([&](const auto& note) { return !generated(note); });
        }
    }
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        bridge->setCandidateSnapshots(normalTimeline, ghostTimeline);
    }
}

void NoaCopilotModel::startGuidePractice()
{
    guideStreamTimer_.stop();
    const bool wasActive = guidePracticeActive_;
    guidePracticeActive_ = false;
    dismissAllActivePreviews();
    warmTimer_.stop();

    guidePracticeActive_ = true;
    guideStage_ = 0;
    guideCandidate_ = 0;
    guideStreamBeat_ = 0;
    guideAcceptedBeats_ = 0;
    const QString initial = document_.chartText();
    lastObservedEditorText_ = initial;
    guideInsertPos_ = initial.size();
    guidePrefixLead_ = (!initial.isEmpty() && !initial.endsWith(QLatin1Char('\n')))
        ? QStringLiteral("\n")
        : QString();

    if (editorBridge_ != nullptr) {
        editorBridge_->setProperty("cursorPosition", guideInsertPos_);
        QTimer::singleShot(0, this, [this]() {
            if (guidePracticeActive_ && guideStage_ == 0 && editorBridge_ != nullptr) {
                editorBridge_->setProperty("cursorPosition", guideInsertPos_);
            }
        });
    }
    preview_.setPositionSeconds(0.0);
    emit ghostStateChanged();
    emit emptyCardChanged();
    emit guidePracticeChanged();
    Q_UNUSED(wasActive);
}

void NoaCopilotModel::stopGuidePractice()
{
    if (!guidePracticeActive_) return;
    guideStreamTimer_.stop();
    guidePracticeActive_ = false;
    guideStage_ = 0;
    guideCandidate_ = 0;
    guideStreamBeat_ = 0;
    guideAcceptedBeats_ = 0;
    selected_ = 0;
    dismissAllActivePreviews();
    preview_.setCopilotReview(false);
    preview_.clearGhostPreviewOverlay();
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        bridge->clearCandidateSnapshots(false);
    }
    emit ghostStateChanged();
    emit emptyCardChanged();
    emit guidePracticeChanged();
}

bool NoaCopilotModel::triggerGuideSymbol(const QString& symbol)
{
    if (!guidePracticeActive_) return false;
    if (symbol == QLatin1String("Tab")) return handleGuideKey(Qt::Key_Tab);
    if (symbol == QLatin1String("↓")) return handleGuideKey(Qt::Key_Down);
    if (symbol == QLatin1String("↑")) return handleGuideKey(Qt::Key_Up);
    if (symbol == QLatin1String("→")) return handleGuideKey(Qt::Key_Right);
    return false;
}

bool NoaCopilotModel::handleGuideKey(int key)
{
    if (!guidePracticeActive_) return false;
    if (key != Qt::Key_Tab && key != Qt::Key_Down && key != Qt::Key_Up && key != Qt::Key_Right) {
        return false;
    }
    if (guideStage_ == 8) {
        return false;
    }
    if (guideStage_ == 1) {
        notifyHint(qtTrId("guide.noa.sandbox.toast.wait_stream"));
        return true;
    }

    if (key == Qt::Key_Tab) {
        if (guideStage_ == 0) {
            guideCandidate_ = 1;
            guideStreamBeat_ = 1;
            guideAcceptedBeats_ = 0;
            guideStage_ = 1;
            showGuideCandidate();
            preview_.setPositionSeconds(2.4);
            guideStreamTimer_.start();
            emit guidePracticeChanged();
            return true;
        }
        if (guideStage_ == 7) {
            const QString remaining = guideCandidateText();
            QScopedValueRollback guard(acceptingStep_, true);
            ghostActive_ = false;
            ghostStreaming_ = false;
            ghostText_.clear();
            ghostReplaceStart_ = 0;
            ghostReplaceEnd_ = 0;
            selected_ = 0;
            updateGhostSplitLines();
            updateEditorGhostStrike();
            applyEditorReplacement(guideInsertPos_, guideInsertPos_, remaining, guideInsertPos_ + remaining.size());
            guideInsertPos_ += remaining.size();
            lastObservedEditorText_ = document_.chartText();
            preview_.setCopilotReview(false);
            preview_.clearGhostPreviewOverlay();
            if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
                bridge->clearCandidateSnapshots(true);
            }
            guideStage_ = 8;
            preview_.setPositionSeconds(2.4);
            emit ghostStateChanged();
            emit guidePracticeChanged();
            return true;
        }
        notifyHint(qtTrId("guide.noa.sandbox.toast.expected").arg(guidePracticeExpectedKey()));
        return true;
    }

    if (key == Qt::Key_Down) {
        if (guideStage_ == 2) {
            guideCandidate_ = 2;
            guideStage_ = 3;
            showGuideCandidate();
            preview_.setPositionSeconds(2.4);
            emit guidePracticeChanged();
            return true;
        }
        if (guideStage_ == 3) {
            guideCandidate_ = 3;
            guideStage_ = 4;
            showGuideCandidate();
            preview_.setPositionSeconds(2.4);
            emit guidePracticeChanged();
            return true;
        }
        notifyHint(qtTrId("guide.noa.sandbox.toast.expected").arg(guidePracticeExpectedKey()));
        return true;
    }

    if (key == Qt::Key_Up) {
        if (guideStage_ == 4) {
            guideCandidate_ = 2;
            guideStage_ = 5;
            showGuideCandidate();
            preview_.setPositionSeconds(2.4);
            emit guidePracticeChanged();
            return true;
        }
        notifyHint(qtTrId("guide.noa.sandbox.toast.expected").arg(guidePracticeExpectedKey()));
        return true;
    }

    if (key == Qt::Key_Right) {
        if (guideStage_ == 5 || guideStage_ == 6) {
            const QString commitText = (guideStage_ == 5)
                ? (guidePrefixLead_ + QStringLiteral("{12}24,3,3,"))
                : QStringLiteral("64,5,5,");
            QScopedValueRollback guard(acceptingStep_, true);
            ghostActive_ = false;
            applyEditorReplacement(guideInsertPos_, guideInsertPos_, commitText, guideInsertPos_ + commitText.size());
            guideInsertPos_ += commitText.size();
            if (guideStage_ == 5) guidePrefixLead_.clear();
            lastObservedEditorText_ = document_.chartText();
            guideAcceptedBeats_ = (guideStage_ == 5) ? 1 : 2;
            guideStage_ = (guideStage_ == 5) ? 6 : 7;
            showGuideCandidate();
            preview_.setPositionSeconds(guideAcceptedBeats_ == 1 ? 2.7 : 3.0);
            emit guidePracticeChanged();
            return true;
        }
        notifyHint(qtTrId("guide.noa.sandbox.toast.expected").arg(guidePracticeExpectedKey()));
        return true;
    }

    return false;
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
    sendSuggestRequest(QStringLiteral("full"), levelOverride, designerOverride);
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
    if (orderedSteps_ && candidates_.isEmpty() && activeRequestMode_ == QStringLiteral("continue")) {
        // The worker can already own an anchor before its first candidate reaches the editor.
        writeDaemonJson({{"op", "discard"}, {"id", nextRpcId_++}});
    }
    activeRequestId_ = 0;
    pendingAcceptAfterStream_ = false;
    if (ghostStreaming_) {
        ghostStreaming_ = false;
        dismissGhost();
    }
    endFullStream();
    discardCandidates();
    emit ghostStateChanged();
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
    const QRegularExpressionMatch match = kBpmRe.match(noaWarmTimingText(document_.chartText()));
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
