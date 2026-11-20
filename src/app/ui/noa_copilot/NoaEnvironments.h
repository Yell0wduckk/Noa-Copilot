#pragma once
#include <QVariantList>

namespace miacode::ui {
QVariantList probeNoaEnvironments(const QString& cpu, const QString& cuda, const QString& distro, bool discover);
}
