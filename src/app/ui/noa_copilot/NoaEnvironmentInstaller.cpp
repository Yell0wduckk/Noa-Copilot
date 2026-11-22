#include "app/ui/noa_copilot/NoaEnvironmentInstaller.h"

#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <QFileInfo>
#include <QProcessEnvironment>

namespace miacode::ui {

NoaEnvironmentInstaller::NoaEnvironmentInstaller(QObject* parent) : QObject(parent)
{
    process_.setProcessChannelMode(QProcess::MergedChannels);
    connect(&process_, &QProcess::readyReadStandardOutput, this, [this] {
        while (process_.canReadLine()) {
            const QString line = QString::fromUtf8(process_.readLine()).trimmed();
            if (line.isEmpty()) continue;
            lastLine_ = line;
            if (line.startsWith("error:", Qt::CaseInsensitive)) processError_ = line;
            if (stage_ == Stage::Packages || stage_ == Stage::Wheels) {
                QString status;
                if (line.startsWith("Progress ")) {
                    const qint64 received = line.section(' ', 1, 1).toLongLong();
                    const qint64 size = line.section(' ', 3, 3).toLongLong();
                    if (size > 0)
                        status = tr("正在下载依赖：%1 / %2 MiB").arg(received / 1048576).arg((size + 1048575) / 1048576);
                } else if (line.startsWith("正在下载 ")) {
                    status = line;
                } else if (line.startsWith("Collecting ")) {
                    status = tr("正在获取 %1").arg(line.mid(11).section(' ', 0, 0));
                } else if (line.startsWith("Installing collected packages")) {
                    status = tr("正在安装已下载的依赖…");
                }
                if (!status.isEmpty() && message_ != status) {
                    message_ = status;
                    emit changed();
                }
            }
        }
    });
    connect(&process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart) finish(process_.errorString());
    });
    connect(&process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus status) {
        if (code == 0 && status == QProcess::NormalExit) advance();
        else if ((stage_ == Stage::Download || stage_ == Stage::Extract) && !pythonUrls_.isEmpty()) downloadPython();
        else if (stage_ == Stage::Repair && !pipIndexes_.isEmpty()) reinstallDamaged();
        else if (stage_ == Stage::Packages && !pipIndexes_.isEmpty()) installPackages();
        else if (!processError_.isEmpty()) finish(processError_);
        else finish(lastLine_.isEmpty() ? tr("安装进程退出 (%1)").arg(code) : lastLine_);
    });
}

NoaEnvironmentInstaller::~NoaEnvironmentInstaller()
{
    process_.disconnect(this);
    if (process_.state() != QProcess::NotRunning) {
        process_.kill();
        process_.waitForFinished();
    }
}

void NoaEnvironmentInstaller::install(const QString& root, bool cuda, const QString& source)
{
    if (busy()) return;
    cuda_ = cuda;
    const QDir environments(QDir(root).filePath("environments"));
    target_ = environments.filePath(cuda ? "torch-2.11.0-cu126" : "torch-2.11.0-cpu");
    const QString bundled = QDir(root).filePath("env");
    if (!cuda && !QFileInfo::exists(target_ + "/python.exe") && QFileInfo::exists(bundled + "/python.exe"))
        target_ = bundled;
    existing_ = QFileInfo::exists(target_ + "/python.exe");
    attemptedRepairs_.clear();
    damaged_.clear();
    wheels_.clear();
    if (!QDir().mkpath(environments.path())) {
        finish(tr("无法创建目录 %1").arg(environments.path()));
        return;
    }
    lock_ = std::make_unique<QLockFile>(environments.filePath(cuda ? ".cuda.lock" : ".cpu.lock"));
    if (!lock_->tryLock()) {
        finish(tr("无法锁定环境目录，若另一个窗口正在安装，请稍后重试"));
        return;
    }
    work_ = std::make_unique<QTemporaryDir>(environments.filePath(".install-XXXXXX"));
    if (!work_->isValid()) {
        finish(work_->errorString());
        return;
    }
    auto env = QProcessEnvironment::systemEnvironment();
    env.insert("TMP", work_->path());
    env.insert("TEMP", work_->path());
    env.insert("PYTHONIOENCODING", "utf-8");
    process_.setProcessEnvironment(env);
    process_.setWorkingDirectory(work_->path());
    pythonUrls_ = {"https://api.nuget.org/v3-flatcontainer/python/3.13.9/python.3.13.9.nupkg",
                   "https://globalcdn.nuget.org/packages/python.3.13.9.nupkg"};
    pipIndexes_.clear();
    wheelUrls_.clear();
    if (source != "official") {
        pythonUrls_.prepend("https://repo.huaweicloud.com/artifactory/api/nuget/v3/nuget-remote/python/3.13.9/python.3.13.9.nupkg");
        const QString tuna = "https://pypi.tuna.tsinghua.edu.cn/simple";
        const QString ustc = "https://mirrors.ustc.edu.cn/pypi/simple";
        const QString aliyun = "https://mirrors.aliyun.com/pypi/simple";
        if (source == "ustc") pipIndexes_ << ustc;
        if (source == "aliyun") pipIndexes_ << aliyun;
        pipIndexes_ << tuna << ustc << aliyun;
        pipIndexes_.removeDuplicates();
        if (source == "aliyun") wheelUrls_ << "https://mirrors.aliyun.com/pytorch-wheels";
        wheelUrls_ << "https://mirror.nju.edu.cn/pytorch/whl" << "https://mirrors.aliyun.com/pytorch-wheels"
                   << "https://mirror.sjtu.edu.cn/pytorch-wheels";
        wheelUrls_.removeDuplicates();
    }
    pipIndexes_ << "https://pypi.org/simple";
    wheelUrls_ << "https://download.pytorch.org/whl";
    sources_ = pipIndexes_;
    if (existing_) probe(Stage::Inspect);
    else {
        wheels_ = {"torch", "torchaudio"};
        downloadPython();
    }
}

QString NoaEnvironmentInstaller::python() const
{
    return existing_ ? target_ + "/python.exe" : work_->filePath("tools/python.exe");
}

void NoaEnvironmentInstaller::probe(Stage stage)
{
    QFile script(":/noa/probe-noa-environment.py");
    if (!script.open(QIODevice::ReadOnly)) {
        finish(script.errorString());
        return;
    }
    QStringList args{"-I", "-X", "utf8", "-u", "-c", QString::fromUtf8(script.readAll())};
    if (cuda_) args << "cuda-build";
    run(stage, tr("正在检查 %1 环境依赖…").arg(cuda_ ? "CUDA" : "CPU"), python(), args);
}

QString NoaEnvironmentInstaller::wheelFile(const QString& name) const
{
    return QString("%1-2.11.0+%2-cp313-cp313-win_amd64.whl").arg(name, cuda_ ? "cu126" : "cpu");
}

void NoaEnvironmentInstaller::downloadWheels()
{
    QFile script(":/noa/download-torch-wheel.py");
    if (!script.open(QIODevice::ReadOnly)) {
        finish(script.errorString());
        return;
    }
    const QStringList args{"-I", "-X", "utf8", "-u", "-c", QString::fromUtf8(script.readAll()),
                           cuda_ ? "cu126" : "cpu", wheels_.join(',')};
    run(Stage::Wheels, tr("正在下载 %1…").arg(wheels_.join(", ")), python(), args + wheelUrls_);
}

void NoaEnvironmentInstaller::downloadPython()
{
    run(Stage::Download, tr("正在下载独立 Python 3.13…"), "curl.exe",
        {"--fail", "--location", "--silent", "--show-error", "--connect-timeout", "12",
         "--retry", "1", "--output", "python.zip", pythonUrls_.takeFirst()});
}

void NoaEnvironmentInstaller::run(Stage stage, const QString& label, const QString& command, const QStringList& args)
{
    stage_ = stage;
    message_ = label;
    lastLine_.clear();
    processError_.clear();
    emit changed();
    process_.start(command, args);
}

void NoaEnvironmentInstaller::advance()
{
    if (stage_ == Stage::Download) {
        run(Stage::Extract, tr("正在解压 Python…"), "tar.exe", {"-xf", "python.zip", "tools"});
    } else if (stage_ == Stage::Extract) {
        downloadWheels();
    } else if (stage_ == Stage::Inspect || stage_ == Stage::Verify) {
        const auto report = QJsonDocument::fromJson(lastLine_.toUtf8()).object().toVariantMap();
        if (!report.contains("repair")) {
            finish(tr("未取得 Python 依赖检测结果：%1").arg(lastLine_));
            return;
        }
        QStringList issues = report.value("chart").toStringList() + report.value("audio").toStringList();
        issues.removeDuplicates();
        if (issues.isEmpty()) {
            if (!existing_ && !QDir().rename(work_->filePath("tools"), target_)) {
                finish(tr("无法将环境安装到 %1").arg(target_));
                return;
            }
            finish();
            return;
        }
        damaged_.clear();
        wheels_.clear();
        pipIndexes_ = sources_;
        const auto repair = report.value("repair").toMap();
        bool newIssue = false;
        for (auto it = repair.cbegin(); it != repair.cend(); ++it) {
            if (attemptedRepairs_.contains(it.key())) continue;
            attemptedRepairs_.insert(it.key());
            newIssue = true;
            const QString version = it.value().toString();
            if (it.key() == "torch" || it.key() == "torchaudio") {
                wheels_ << it.key();
                if (!version.isEmpty()) damaged_ << wheelFile(it.key());
            } else if (!version.isEmpty()) {
                damaged_ << it.key() + "==" + version;
            }
        }
        if (!newIssue) {
            finish(issues.join('\n'));
            return;
        }
        if (existing_ && stage_ == Stage::Inspect) emit repairing(python());
        if (!wheels_.isEmpty()) downloadWheels();
        else if (!damaged_.isEmpty()) reinstallDamaged();
        else installPackages();
    } else if (stage_ == Stage::Wheels && !damaged_.isEmpty()) {
        reinstallDamaged();
    } else if (stage_ == Stage::Repair || stage_ == Stage::Wheels) {
        pipIndexes_ = sources_;
        installPackages();
    } else if (stage_ == Stage::Packages) {
        probe(Stage::Verify);
    }
}

static QStringList pipInstallArgs(const QString& index, const QString& cache)
{
    return {"-I", "-X", "utf8", "-u", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
            "install", "--break-system-packages", "--cache-dir", cache, "--retries", "1", "--timeout", "15",
            "--no-warn-script-location", "--progress-bar", "raw", "--index-url", index};
}

void NoaEnvironmentInstaller::reinstallDamaged()
{
    const QString index = pipIndexes_.takeFirst();
    auto args = pipInstallArgs(index, work_->filePath("pip-cache"));
    args << "--force-reinstall" << "--no-deps";
    run(Stage::Repair, tr("正在修复 %1").arg(damaged_.join(", ")), python(), args + damaged_);
}

void NoaEnvironmentInstaller::installPackages()
{
    const QString index = pipIndexes_.takeFirst();
    auto args = pipInstallArgs(index, work_->filePath("pip-cache"));
    for (const QString& name : {QStringLiteral("torch"), QStringLiteral("torchaudio")})
        args << (wheels_.contains(name) ? wheelFile(name) : name + "==2.11.0");
    args << "numpy==2.1.3" << "scipy" << "soundfile" << "transformers==4.47.1"
         << "huggingface-hub==0.36.2" << "safetensors" << "einops";
    run(Stage::Packages, tr("正在安装依赖（%1）").arg(index), python(), args);
}

void NoaEnvironmentInstaller::finish(const QString& error)
{
    work_.reset();
    lock_.reset();
    stage_ = Stage::Idle;
    message_ = error.isEmpty()
        ? tr("%1 环境已就绪").arg(cuda_ ? "CUDA 12.6" : "CPU")
        : tr("%1 环境安装失败：%2").arg(cuda_ ? "CUDA" : "CPU", error);
    emit changed();
    if (error.isEmpty()) emit installed(target_ + "/python.exe", cuda_);
}

}
