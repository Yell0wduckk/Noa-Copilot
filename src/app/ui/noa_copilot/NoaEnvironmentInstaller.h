#pragma once

#include <QObject>
#include <QLockFile>
#include <QProcess>
#include <QSet>
#include <QTemporaryDir>
#include <memory>

namespace miacode::ui {

class NoaEnvironmentInstaller final : public QObject
{
    Q_OBJECT
public:
    explicit NoaEnvironmentInstaller(QObject* parent = nullptr);
    ~NoaEnvironmentInstaller() override;
    void install(const QString& root, bool cuda, const QString& source);
    bool busy() const { return stage_ != Stage::Idle; }
    QString message() const { return message_; }

signals:
    void changed();
    void repairing(const QString& python);
    void installed(const QString& python, bool cuda);

private:
    enum class Stage { Idle, Inspect, Download, Extract, Wheels, Repair, Packages, Verify };
    void downloadPython();
    void probe(Stage stage);
    void downloadWheels();
    void reinstallDamaged();
    QString wheelFile(const QString& name) const;
    QString python() const;
    void installPackages();
    void run(Stage stage, const QString& label, const QString& command, const QStringList& args);
    void advance();
    void finish(const QString& error = {});

    QProcess process_;
    std::unique_ptr<QTemporaryDir> work_;
    std::unique_ptr<QLockFile> lock_;
    QStringList wheels_;
    QStringList damaged_;
    QSet<QString> attemptedRepairs_;
    QStringList sources_;
    bool existing_ = false;
    Stage stage_ = Stage::Idle;
    bool cuda_ = false;
    QString target_;
    QStringList pythonUrls_;
    QStringList wheelUrls_;
    QStringList pipIndexes_;
    QString message_;
    QString lastLine_;
    QString processError_;
};
}
