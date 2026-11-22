#include "app/ui/noa_copilot/NoaEnvironments.h"
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLibrary>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>
#include <QStandardPaths>
#include <future>
#include <vector>
#include <QThreadPool>
#include <QtConcurrent/QtConcurrentRun>

namespace miacode::ui {
namespace {
QString run(const QString& program, const QStringList& args, int seconds = 20)
{
    QProcess child;
    child.setProcessChannelMode(QProcess::MergedChannels);
    child.start(program, args);
    if (!child.waitForFinished(seconds * 1000)) {
        const QString failure = child.errorString();
        child.kill();
        child.waitForFinished();
        return failure;
    }
    return QString::fromUtf8(child.readAll()).trimmed();
}
QStringList lines(const QString& file)
{
    QFile f(file);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    return QString::fromUtf8(f.readAll()).split(QRegularExpression("[\\r\\n]+"), Qt::SkipEmptyParts);
}
}

QVariantList probeNoaEnvironments(const QString& cpu, const QString& cuda, const QString& distro, bool discover)
{
    QSet<QString> paths;
    auto add = [&](QString path) {
        if (!path.trimmed().isEmpty())
            paths.insert(QDir::fromNativeSeparators(path.trimmed()));
    };
    add(cpu);
    add(cuda);
    const QString home = QDir::homePath();
    QStringList wsl;
    if (!distro.isEmpty())
        wsl << "-d" << distro;
#ifdef Q_OS_WIN
    const bool driver = QLibrary("nvcuda").load() || !QStandardPaths::findExecutable("nvidia-smi").isEmpty();
#else
    const bool driver = !QStandardPaths::findExecutable("nvidia-smi").isEmpty();
#endif
    if (discover) {
        auto py = std::async(std::launch::async, [] { return run("py", { "-0p" }, 5); });
        auto linuxPaths = std::async(std::launch::async, [wsl] {
#ifdef Q_OS_WIN
            return run("wsl.exe",
                wsl
                    + QStringList { "--exec", "sh", "-c",
                        "for p in /usr/bin/python3 /usr/local/bin/python* ~/.venvs/*/bin/python "
                        "~/.venvs/*/*/bin/python ~/.local/share/uv/python/*/bin/python* "
                        "~/*conda*/envs/*/bin/python; do "
                        "test -x \"$p\" || continue; root=$(dirname \"$(dirname \"$p\")\"); "
                        "for f in \"$root\"/lib/python*/site-packages/torch/version.py; do test -f \"$f\" && echo "
                        "\"$p\"; done; done" },
                10);
#else
            return QString();
#endif
        });
        for (auto prefix : lines(home + "/.conda/environments.txt"))
            add(prefix + "/python.exe");
        for (auto dir : QProcessEnvironment::systemEnvironment().value("PATH").split(QDir::listSeparator())) {
            if (QFileInfo::exists(dir + "/python.exe"))
                add(dir + "/python.exe");
        }
        for (auto base : { home + "/.venvs", home + "/.virtualenvs", home + "/.local/share/uv/python",
                 qEnvironmentVariable("APPDATA") + "/uv/python", home + "/anaconda3/envs" }) {
            for (auto dir : QDir(base).entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
                for (auto exe : { "/Scripts/python.exe", "/python.exe", "/bin/python" })
                    if (QFileInfo::exists(base + '/' + dir + exe))
                        add(base + '/' + dir + exe);
            }
        }
        QRegularExpression executable(
            R"(([A-Za-z]:[^\r\n]*?python[^\\/\r\n]*\.exe))", QRegularExpression::CaseInsensitiveOption);
        auto matches = executable.globalMatch(py.get());
        while (matches.hasNext())
            add(matches.next().captured(1));
        for (auto path : linuxPaths.get().split('\n'))
            if (path.startsWith('/'))
                add(path);
    }
    QThreadPool pool;
    pool.setMaxThreadCount(3);
    std::vector<QFuture<QVariantMap>> probes;
    QStringList ordered = paths.values();
    ordered.sort(Qt::CaseInsensitive);
    QFile script(":/noa/probe-noa-environment.py");
    script.open(QIODevice::ReadOnly);
    const QString probe = QString::fromUtf8(script.readAll());
    for (const QString& path : ordered) {
#ifdef Q_OS_WIN
        bool isWsl = path.startsWith('/');
#else
        bool isWsl = false;
#endif
        bool cudaBuild = true;
        if (!isWsl) {
            QFileInfo exe(path);
            if (!exe.isFile())
                continue;
            QString base = exe.absolutePath();
            if (base.endsWith("/Scripts", Qt::CaseInsensitive))
                base = QFileInfo(base).absolutePath();
            QString version = base + "/Lib/site-packages/torch/version.py";
            if (!QFileInfo::exists(version)) {
                const QStringList config = lines(base + "/pyvenv.cfg");
                for (const auto& line : config)
                    if (line.startsWith("home = ")) {
                        const QString candidate = line.mid(7) + "/Lib/site-packages/torch/version.py";
                        if (QFileInfo::exists(candidate))
                            version = candidate;
                    }
            }
            QFile info(version);
            if (info.open(QIODevice::ReadOnly)) {
                static const QRegularExpression compiled(R"(cuda[^=\n]*=\s*['"][0-9])");
                cudaBuild = compiled.match(QString::fromUtf8(info.readAll())).hasMatch();
            } else if (path != QDir::fromNativeSeparators(cpu) && path != QDir::fromNativeSeparators(cuda))
                continue;
        }
        probes.push_back(QtConcurrent::run(&pool, [path, probe, isWsl, wsl, driver, cudaBuild] {
            QStringList args {"-X", "utf8", "-c", probe};
            if (driver && cudaBuild) args << "cuda";
            const QString result = isWsl ? run("wsl.exe", wsl + QStringList {"--exec", path} + args, 90)
                                         : run(path, args, 90);
            const auto obj = QJsonDocument::fromJson(result.section('\n', -1).toUtf8()).object().toVariantMap();
            QStringList issues = obj.value("chart").toStringList();
            if (!obj.contains("chart"))
                issues << (result.isEmpty() ? QStringLiteral("Python 未返回检测结果") : result);
            const bool cpuOk = issues.isEmpty();
            const QStringList audio = obj.value("audio").toStringList();
            for (const QString& item : audio)
                if (!issues.contains(item)) issues << item;
            const QString gpu = obj.value("gpu").toString();
            QString cudaReason = obj.value("cuda_error").toString();
            if (cudaReason.isEmpty() && gpu.isEmpty() && cpuOk)
                cudaReason = driver ? QStringLiteral("此环境没有可用的 CUDA") : QStringLiteral("未检测到 NVIDIA 驱动");
            return QVariantMap {{"path", path}, {"cpu", cpuOk}, {"cuda", cpuOk && !gpu.isEmpty()},
                {"mert", audio.isEmpty() && cpuOk}, {"reason", issues.join('\n')}, {"cudaReason", cudaReason},
                {"label", cpuOk ? path + " · " + obj.value("version").toString() + " " + gpu : path}};
        }));
    }
    QVariantList rows;
    for (const auto& result : probes) rows.append(result.result());
    return rows;
}
}
