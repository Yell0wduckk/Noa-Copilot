#pragma once

#include <QString>

namespace miacode::ui::noa_cache {
struct Usage {
    qint64 bytes = 0;
    int removed = 0;
    int failed = 0;
};
void rememberProject(const QString& directory);
Usage inspect(bool clear);
}
