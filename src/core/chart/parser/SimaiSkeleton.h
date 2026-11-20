#pragma once
#include <QStringView>

namespace miacode::simai {
inline bool isSkeletonItem(QStringView text)
{
    if (text.isEmpty())
        return false;
    if (text.front().toLower() == 'x') {
        text = text.sliced(1);
        if (text.isEmpty())
            return true;
    }
    for (QChar c : text)
        if (c != '*')
            return false;
    return true;
}
}
