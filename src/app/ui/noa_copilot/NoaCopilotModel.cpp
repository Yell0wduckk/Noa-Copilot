#include "app/ui/noa_copilot/NoaCopilotModel.h"

#include "app/services/PreferenceDocument.h"
#include "app/services/UiRequestService.h"
#include "app/ui/document/DocumentModel.h"
#include "app/ui/preview/PreviewModel.h"
#include "app/ui/timeline/TimelineModel.h"
#include "common/DebugLog.h"
#include "core/chart/ChartAssetPaths.h"
#include "core/chart/parser/SimaiParser.h"
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
    debounceTimer_.setSingleShot(true);
    connect(&debounceTimer_, &QTimer::timeout, this, [this]() {
        if (!enabled_ || !autoContinue_ || !hasCheckpoint() || busy() || inlineCardVisible_ || pendingRangeActive_) {
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
        releaseVram();
    });

    // Ensure default checkpoints directory exists so users can drop .ckpt files into it.
    QDir().mkpath(defaultCheckpointsDir());
    QDir().mkpath(resolvedCacheDir());

    rebuildDesignerOptions({});
    loadPreferences();
    refreshCheckpoints();
    refreshAudioCacheSize();

    connect(&document_, &DocumentModel::documentStateChanged, this, [this]() {
        dismissGhost();
        if (pendingRangeActive_) {
            rejectPendingRange();
        }
        sendBindSongIfNeeded();
        emit emptyCardChanged();
    });
    connect(&document_, &DocumentModel::currentDifficultyChanged, this, [this]() {
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
        emit emptyCardChanged();
    });

    if (enabled_) {
        QTimer::singleShot(300, this, [this]() {
            ensureDaemonRunning();
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
    if (pythonMode_ == QStringLiteral("custom") && !customPythonPath_.trimmed().isEmpty()) {
        return QDir::cleanPath(customPythonPath_.trimmed());
    }
    if (usesWsl()) return wslPython_;
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

QVariantList NoaCopilotModel::creativityPresetOptions() const
{
    return {
        QVariantMap{{QStringLiteral("value"), QStringLiteral("stable")}, {QStringLiteral("label"), QStringLiteral("稳健")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("balanced")}, {QStringLiteral("label"), QStringLiteral("均衡")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("creative")}, {QStringLiteral("label"), QStringLiteral("灵动")}},
    };
}

QVariantList NoaCopilotModel::deviceModeOptions() const
{
    return {
        QVariantMap{{QStringLiteral("value"), QStringLiteral("auto")}, {QStringLiteral("label"), QStringLiteral("自动（所选 Python 环境）")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("cuda")}, {QStringLiteral("label"), QStringLiteral("CUDA（内置模式使用 WSL）")}},
        QVariantMap{{QStringLiteral("value"), QStringLiteral("cpu")}, {QStringLiteral("label"), QStringLiteral("CPU")}},
    };
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
    enabled_ = sec.value(QStringLiteral("enabled")).toBool(true);
    autoContinue_ = sec.value(QStringLiteral("auto_continue")).toBool(true);
    creativityPreset_ = sec.value(QStringLiteral("creativity_preset")).toString(QStringLiteral("balanced"));
    defaultDesigner_ = sec.value(QStringLiteral("default_designer")).toString();
    device_ = sec.value(QStringLiteral("device")).toString(QStringLiteral("auto"));
    precision_ = sec.value(QStringLiteral("precision")).toString(QStringLiteral("bf16"));
    temperature_ = sec.value(QStringLiteral("temperature")).toDouble(1.0);
    slideBias_ = sec.value(QStringLiteral("slide_bias")).toDouble(0.0);
    convEnabled_ = sec.value(QStringLiteral("conv_enabled")).toBool(true);
    mertEnabled_ = sec.value(QStringLiteral("mert_enabled")).toBool(true);
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
    // The previous Windows bundle saved its default interpreter in user preferences.
    if (sourceMode_ == QStringLiteral("bundled") && pythonMode_ == QStringLiteral("bundled")
        && !bundledPython.isEmpty() && wslDistribution_ == defaults.value(QStringLiteral("distribution")).toString()
        && (wslPython_.isEmpty() || wslPython_ == defaults.value(QStringLiteral("previous_python")).toString())) {
        wslPython_ = bundledPython;
    }

    if (creativityPreset_ != QStringLiteral("custom")) {
        applyCreativityPresetValues(creativityPreset_);
    }
}

void NoaCopilotModel::savePreferences() const
{
    QJsonObject root = PreferenceDocument::loadPreferencesObject();
    QJsonObject sec;
    sec.insert(QStringLiteral("enabled"), enabled_);
    sec.insert(QStringLiteral("auto_continue"), autoContinue_);
    sec.insert(QStringLiteral("creativity_preset"), creativityPreset_);
    sec.insert(QStringLiteral("default_designer"), defaultDesigner_);
    sec.insert(QStringLiteral("device"), device_);
    sec.insert(QStringLiteral("precision"), precision_);
    sec.insert(QStringLiteral("temperature"), temperature_);
    sec.insert(QStringLiteral("slide_bias"), slideBias_);
    sec.insert(QStringLiteral("conv_enabled"), convEnabled_);
    sec.insert(QStringLiteral("mert_enabled"), mertEnabled_);
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

void NoaCopilotModel::applyCreativityPresetValues(const QString& preset)
{
    if (preset == QStringLiteral("stable")) {
        temperature_ = 0.85;
    } else if (preset == QStringLiteral("creative")) {
        temperature_ = 1.12;
    } else if (preset == QStringLiteral("balanced")) {
        temperature_ = 1.00;
    }
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

void NoaCopilotModel::setCreativityPreset(const QString& preset)
{
    if (creativityPreset_ == preset) {
        return;
    }
    creativityPreset_ = preset;
    applyCreativityPresetValues(preset);
    savePreferences();
    emit configChanged();
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
    const bool wasWsl = usesWsl();
    device_ = value;
    savePreferences();
    emit configChanged();
    if (enabled_) {
        if (usesWsl() != wasWsl) restartDaemon();
        else sendConfigureRequest();
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

void NoaCopilotModel::setConvEnabled(bool enabled)
{
    if (convEnabled_ == enabled) return;
    convEnabled_ = enabled;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setMertEnabled(bool enabled)
{
    if (mertEnabled_ == enabled) return;
    mertEnabled_ = enabled;
    savePreferences();
    emit configChanged();
}

void NoaCopilotModel::setTemperature(double value)
{
    value = qBound(0.0, value, 2.0);
    if (qFuzzyCompare(temperature_, value)) {
        return;
    }
    temperature_ = value;
    creativityPreset_ = QStringLiteral("custom");
    savePreferences();
    emit configChanged();
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
    return pythonMode_ == QStringLiteral("bundled") && device_ == QStringLiteral("cuda");
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
    if (enabled_ && usesWsl()) restartDaemon();
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
    QDirIterator it(resolvedCacheDir(), QDir::Files, QDirIterator::Subdirectories);
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
    QDir cacheDir(resolvedCacheDir());
    if (cacheDir.exists()) {
        cacheDir.removeRecursively();
    }
    QDir().mkpath(resolvedCacheDir());
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
               << QStringLiteral("PYTHONNOUSERSITE=1") << wslPython_;
        daemonProcess_->start(QStringLiteral("wsl.exe"), launch + args);
    } else {
        daemonProcess_->start(pythonExe, args);
    }
}

void NoaCopilotModel::stopDaemon()
{
    endFullStream();
    stdoutBuffer_.clear();
    debounceTimer_.stop();
    idleVramTimer_.stop();
    activeRequestId_ = 0;
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
    if (!enabled_ || !daemonConfigured_ || !checkpointLoadedInDaemon_ || !document_.hasDocument()) {
        return;
    }
    const QString audioPath = miacode::chart_assets::resolveTrackPath(document_.currentFilePath());
    if (audioPath.isEmpty() || !QFileInfo::exists(audioPath)) {
        return;
    }
    const double firstSec = currentChartFirstSeconds();
    const double bpm = currentChartBpm();
    const QString songKey = QStringLiteral("%1|%2|%3|%4")
                                .arg(audioPath)
                                .arg(firstSec, 0, 'f', 4)
                                .arg(bpm, 0, 'f', 4)
                                .arg(checkpointPath_);
    if (boundSongKey_ == songKey) {
        return;
    }
    boundSongKey_ = songKey;
    QJsonObject req;
    req.insert(QStringLiteral("op"), QStringLiteral("bind_song"));
    req.insert(QStringLiteral("id"), nextRpcId_++);
    req.insert(QStringLiteral("song_id"), songKey);
    req.insert(QStringLiteral("audio"), audioPath);
    req.insert(QStringLiteral("first"), firstSec);
    req.insert(QStringLiteral("bpm"), bpm);
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
        updateStatus(QStringLiteral("generating"), QStringLiteral("正在流式生成整曲..."));
    } else if (mode == QStringLiteral("range")) {
        updateStatus(QStringLiteral("generating"), QStringLiteral("正在生成段落..."));
    } else {
        updateStatus(QStringLiteral("generating"), QStringLiteral("正在续写..."));
    }

    const QString audioPath = miacode::chart_assets::resolveTrackPath(document_.currentFilePath());
    const double levelVal = !levelOverride.trimmed().isEmpty()
        ? parseLevelNumeric(levelOverride)
        : parseLevelNumeric(document_.currentDifficultyLevel());
    const QString designerVal = effectiveDesigner(designerOverride);

    QJsonObject req;
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
    req.insert(QStringLiteral("conv_enabled"), convEnabled_);
    req.insert(QStringLiteral("mert_enabled"), mertEnabled_);
    req.insert(QStringLiteral("seed"), seed_);
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

void NoaCopilotModel::handleDaemonMessage(const QJsonObject& message)
{
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
        }
        return;
    }

    if (event == QStringLiteral("status")) {
        QString st = message.value(QStringLiteral("state")).toString(QStringLiteral("warming_audio"));
        if (st == QStringLiteral("encoding_audio")) {
            st = QStringLiteral("warming_audio");
        }
        const QString msg = message.value(QStringLiteral("message")).toString(QStringLiteral("正在处理..."));
        updateStatus(st, msg);
        return;
    }

    if (event == QStringLiteral("song_bound")) {
        refreshAudioCacheSize();
        if (activeRequestId_ == 0 && checkpointLoadedInDaemon_) {
            updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
        }
        return;
    }

    if (event == QStringLiteral("stream_chunk")) {
        const int id = message.value(QStringLiteral("id")).toInt();
        if (id != activeRequestId_) {
            return;
        }
        streamingBeat_ = message.value(QStringLiteral("beat")).toInt();
        streamingTotalBeats_ = message.value(QStringLiteral("total_beats")).toInt();
        streamingProgress_ = message.value(QStringLiteral("progress")).toDouble();
        emit streamingProgressChanged();
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
            endFullStream();
            updateStatus(QStringLiteral("error"), QStringLiteral("生成失败"), err);
            return;
        }

        updateStatus(QStringLiteral("ready"), QStringLiteral("就绪"), activeCheckpointName());
        refreshAudioCacheSize();

        if (activeRequestRevision_ != document_.documentRevision()) {
            activeRequestId_ = 0;
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
            endFullStream();
            return;
        }

        activeRequestId_ = 0;
        if (mode == QStringLiteral("range")) {
            if (text.trimmed().isEmpty()) {
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
            applyDualSurfaceGhostPreview(ghostSimai, startSec, activeBpm);
            return;
        }

        // mode == "continue"
        if (text.trimmed().isEmpty()) {
            return;
        }
        // Verify document hasn't moved away while the suggestion was in flight
        if (static_cast<quint64>(message.value(QStringLiteral("revision")).toInteger()) != document_.documentRevision()) {
            return;
        }
        ghostActive_ = true;
        ghostText_ = text;
        ghostReplaceStart_ = replaceStart;
        ghostReplaceEnd_ = replaceEnd;
        ghostSteps_ = message.value(QStringLiteral("steps")).toArray();
        ghostCurrentStep_ = 0;
        updateGhostSplitLines();
        emit ghostStateChanged();
        applyDualSurfaceGhostPreview(ghostSimai, startSec, activeBpm);
        return;
    }

    if (event == QStringLiteral("cancelled")) {
        const int id = message.value(QStringLiteral("id")).toInt();
        if (id == activeRequestId_ || id <= 0) {
            activeRequestId_ = 0;
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
    if (!document_.applyGeneratedChart(text)) {
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
    double activeBpm)
{
    if (ghostSimai.trimmed().isEmpty()) {
        clearDualSurfaceGhostPreview();
        return;
    }
    QString normalizedSimai = ghostSimai;
    if (!normalizedSimai.trimmed().startsWith(QLatin1Char('('))) {
        normalizedSimai.prepend(QStringLiteral("(%1)").arg(activeBpm > 0.0 ? activeBpm : 160.0));
    }
    if (!normalizedSimai.trimmed().endsWith(QLatin1Char('E'), Qt::CaseInsensitive)) {
        normalizedSimai.append(QStringLiteral("\nE"));
    }

    // 1. Circular Preview Ghost Notes
    if (auto* runtime = qobject_cast<PreviewRuntime*>(preview_.runtime())) {
        SimaiParseResult parsed = SimaiParser::parseForTimeline(normalizedSimai);
        for (TimelineNoteMarker& marker : parsed.noteMarkers) {
            marker.second += startSecond;
            if (marker.endSecond >= 0.0) {
                marker.endSecond += startSecond;
            }
            if (marker.slideTraceSecond >= 0.0) {
                marker.slideTraceSecond += startSecond;
            }
            if (marker.availableSecond >= 0.0) {
                marker.availableSecond += startSecond;
            }
            for (double& shootSec : marker.slideSegmentShootSeconds) {
                shootSec += startSecond;
            }
        }
        runtime->setGhostNoteMarkers(parsed.noteMarkers);
    }

    // 2. Bottom Timeline Ghost Notes
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
        TimelineQuickModel ghostModel;
        ghostModel.rebuildFromText(normalizedSimai, startSecond, {});
        bridge->setGhostSnapshot(ghostModel.snapshot());
    }
}

void NoaCopilotModel::clearDualSurfaceGhostPreview()
{
    if (auto* runtime = qobject_cast<PreviewRuntime*>(preview_.runtime())) {
        runtime->setGhostNoteMarkers({});
    }
    if (auto* bridge = qobject_cast<TimelineQuickStateBridge*>(timeline_.stateBridge())) {
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

void NoaCopilotModel::bindEditor(QObject* editorObject)
{
    editorBridge_ = editorObject;
}

void NoaCopilotModel::notifyEditorCursorOrTextChanged(
    int cursorPos,
    int selStart,
    int selEnd,
    bool textChanged)
{
    lastCursorPos_ = cursorPos;
    lastSelStart_ = selStart;
    lastSelEnd_ = selEnd;
    idleVramTimer_.start();

    if (ghostActive_) {
        dismissGhost();
    }
    if (textChanged && !fullStreamingActive_) {
        if (pendingRangeActive_) rejectPendingRange();
        sendSyncChartIfNeeded();
        if (enabled_ && autoContinue_ && hasCheckpoint() && !inlineCardVisible_ && !pendingRangeActive_) {
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
        if (ghostActive_) {
            dismissGhost();
        }
        return;
    }
    notifyEditorCursorOrTextChanged(cursorPos, selStart, selEnd, textChanged);
}

bool NoaCopilotModel::handleEditorKeyPress(int key, int modifiers)
{
    const bool ctrl = (modifiers & Qt::ControlModifier) != 0;
    const bool shift = (modifiers & Qt::ShiftModifier) != 0;
    const bool alt = (modifiers & Qt::AltModifier) != 0;

    if (pendingRangeActive_) {
        if (key == Qt::Key_Tab && !ctrl && !shift && !alt) {
            return acceptPendingRange();
        }
        if (key == Qt::Key_Escape) {
            rejectPendingRange();
            return true;
        }
        if (key == Qt::Key_R && !ctrl && !alt) {
            regeneratePendingRange();
            return true;
        }
        if (key == Qt::Key_Space && !ctrl && !alt) {
            auditionPendingRange();
            return true;
        }
    }

    if (ghostActive_) {
        if (key == Qt::Key_Tab && !ctrl && !shift && !alt) {
            return acceptFullGhost();
        }
        if (key == Qt::Key_Right && ctrl && !shift && !alt) {
            return acceptNextGhostStep();
        }
        if (key == Qt::Key_Escape) {
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

    if (fullStreamingActive_ && key == Qt::Key_Escape) {
        cancelActiveGeneration();
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
    if (!ghostActive_ || ghostText_.isEmpty()) {
        return false;
    }
    const QString textToInsert = ghostText_;
    const int startPos = ghostReplaceStart_;
    const int endPos = ghostReplaceEnd_;
    dismissGhost();
    return applyEditorReplacement(startPos, endPos, textToInsert);
}

bool NoaCopilotModel::acceptNextGhostStep()
{
    if (!ghostActive_ || ghostText_.isEmpty()) {
        return false;
    }
    if (ghostSteps_.isEmpty() || ghostCurrentStep_ >= ghostSteps_.size()) {
        return acceptFullGhost();
    }

    const QJsonObject stepObj = ghostSteps_.at(ghostCurrentStep_).toObject();
    const QString stepText = stepObj.value(QStringLiteral("accept_text")).toString();
    const QString remText = stepObj.value(QStringLiteral("preview_remaining_text")).toString();
    const QString remGhostSimai = stepObj.value(QStringLiteral("remaining_ghost_simai")).toString();
    const double remStartSec = stepObj.value(QStringLiteral("remaining_start_second")).toDouble();
    const double remBpm = stepObj.value(QStringLiteral("remaining_bpm")).toDouble(currentChartBpm());

    if (stepText.isEmpty()) {
        return acceptFullGhost();
    }

    const int insertStart = ghostReplaceStart_;
    const int insertEnd = ghostReplaceEnd_;
    const QJsonArray savedSteps = ghostSteps_;
    const int nextStepIdx = ghostCurrentStep_ + 1;

    ghostActive_ = false;
    applyEditorReplacement(insertStart, insertEnd, stepText);

    if (remText.trimmed().isEmpty() || nextStepIdx >= savedSteps.size()) {
        dismissGhost();
        return true;
    }

    ghostActive_ = true;
    ghostText_ = remText;
    ghostReplaceStart_ = insertStart + stepText.size();
    ghostReplaceEnd_ = ghostReplaceStart_;
    ghostSteps_ = savedSteps;
    ghostCurrentStep_ = nextStepIdx;
    updateGhostSplitLines();
    emit ghostStateChanged();
    applyDualSurfaceGhostPreview(remGhostSimai, remStartSec, remBpm);
    return true;
}

void NoaCopilotModel::dismissGhost()
{
    if (!ghostActive_ && ghostText_.isEmpty()) {
        return;
    }
    ghostActive_ = false;
    ghostText_.clear();
    ghostFirstLine_.clear();
    ghostExtraLines_.clear();
    ghostSteps_ = QJsonArray();
    ghostCurrentStep_ = 0;
    emit ghostStateChanged();
    if (!pendingRangeActive_ && !fullStreamingActive_) {
        clearDualSurfaceGhostPreview();
    }
}

void NoaCopilotModel::openInlineCard()
{
    if (!enabled_ || !document_.hasDocument() || document_.currentDifficultyId() <= 0) {
        return;
    }
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
    emit pendingRangeChanged();
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
    if (!pendingRangeActive_) {
        return;
    }
    pendingRangeActive_ = false;
    pendingRangeText_.clear();
    emit pendingRangeChanged();
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
    endFullStream();
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
