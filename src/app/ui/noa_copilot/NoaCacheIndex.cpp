#include "app/ui/noa_copilot/NoaCacheIndex.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QRegularExpression>
#include <QStandardPaths>
#include <filesystem>

namespace miacode::ui::noa_cache {
namespace {
namespace fs = std::filesystem;

QSettings indexFile()
{
    return QSettings(QDir(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation))
        .filePath("noacopilot-cache-index.ini"), QSettings::IniFormat);
}

bool missing(const std::error_code& error)
{
    return error == std::errc::no_such_file_or_directory || error == std::errc::not_a_directory;
}

bool linked(const QFileInfo& info)
{
    return info.isSymLink() || info.isJunction();
}

struct Inventory {
    Usage usage;
    QVariantMap prior;
    QVariantMap kept;

    void unreadable(const QString& name)
    {
        ++usage.failed;
        const auto prefix = name + '/';
        for (auto it = prior.cbegin(); it != prior.cend(); ++it)
            if (name.isEmpty() || it.key() == name || it.key().startsWith(prefix))
                kept.insert(it.key(), it.value());
    }
};

bool cacheEntry(const QString& name)
{
    static const QRegularExpression file(
        QStringLiteral("^(?:[a-fA-F0-9]{64}|(?:mert|roformer|drafter)_[^/]+)\\.pt$"));
    return file.match(name).hasMatch() || name == "manifest.json" || name == "huggingface"
        || name == "modules" || name == "torch" || name == "triton" || name == "cuda";
}

// Each root is reserved for Noa. Previously measured sizes are only used for an
// inaccessible path, never as deletion targets or as evidence that a file still exists.
void visit(const fs::path& path, const QString& name, bool clear, Inventory& inventory)
{
    std::error_code error;
    const auto status = fs::symlink_status(path, error);
    if (missing(error) || (!error && !fs::exists(status))) return;
    if (error || linked(QFileInfo(path))) {
        inventory.unreadable(name);
        return;
    }
    if (fs::is_directory(status)) {
        fs::directory_iterator it(path, error), end;
        while (!error && it != end) {
            const QString leaf = QFileInfo(it->path()).fileName();
            // Ignore unrelated files and temporary, unpublished tensors from another editor.
            if (!name.isEmpty() || cacheEntry(leaf))
                visit(it->path(), name.isEmpty() ? leaf : name + '/' + leaf, clear, inventory);
            it.increment(error);
        }
        if (error && !missing(error)) inventory.unreadable(name);
        if (clear && !name.isEmpty() && !error && fs::is_empty(path, error) && !error) {
            fs::remove(path, error);
            if (error && !missing(error)) ++inventory.usage.failed;
        }
        return;
    }
    if (name.isEmpty() || !fs::is_regular_file(status)) {
        inventory.unreadable(name);
        return;
    }
    if (clear) {
        if (fs::remove(path, error)) {
            inventory.prior.remove(name);
            ++inventory.usage.removed;
            return;
        }
        if (!error || missing(error)) return;
        ++inventory.usage.failed;
    }
    const auto size = fs::file_size(path, error);
    if (!error) inventory.kept.insert(name, static_cast<qint64>(size));
    else if (!missing(error) && inventory.prior.contains(name))
        inventory.kept.insert(name, inventory.prior.value(name));
}

Usage inspectDirectory(const QString& path, bool clear, QSettings& index, const QString& key)
{
    const auto sizeKey = "sizes/" + key;
    const auto prior = index.value(sizeKey).toMap();
    Inventory inventory{{}, prior, {}};
    if (linked(QFileInfo(QFileInfo(path).absolutePath()))) inventory.unreadable({});
    else visit(QFileInfo(path).filesystemAbsoluteFilePath(), {}, clear, inventory);
    for (const auto& size : inventory.kept) inventory.usage.bytes += size.toLongLong();
    if (inventory.kept != prior) {
        if (inventory.kept.isEmpty()) index.remove(sizeKey);
        else index.setValue(sizeKey, inventory.kept);
    }
    return inventory.usage;
}
}

void rememberProject(const QString& directory)
{
    if (directory.isEmpty()) return;
    const QString path = QDir::cleanPath(QFileInfo(directory).absoluteFilePath());
    QString identity = path;
#ifdef Q_OS_WIN
    identity = identity.toCaseFolded();
#endif
    const QString key = QString::fromLatin1(QCryptographicHash::hash(identity.toUtf8(), QCryptographicHash::Sha256).toHex());
    auto index = indexFile();
    if (!index.contains(key)) index.setValue(key, path);
}

Usage inspect(bool clear)
{
    Usage total;
    auto index = indexFile();
    const auto add = [&total](const Usage& entry) {
        total.bytes += entry.bytes;
        total.failed += entry.failed;
        total.removed += entry.removed;
    };
    for (const auto& key : index.childKeys()) {
        const QString project = index.value(key).toString();
        if (project.isEmpty() || !QDir::isAbsolutePath(project)) continue;
        const auto path = QDir(project).filePath(".miacode/.noacopilot");
        const auto entry = inspectDirectory(path, clear, index, key);
        add(entry);
        if (clear && !entry.failed && !entry.bytes) index.remove(key);
    }
    // Older Noa releases used this dedicated directory for features and Transformers modules.
    const auto local = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    if (!local.isEmpty()) add(inspectDirectory(QDir(local).filePath("noa-copilot/cache"), clear, index, "legacy"));
    return total;
}
}
