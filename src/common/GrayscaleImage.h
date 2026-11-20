#pragma once

#include <QImage>

namespace miacode {

inline QColor grayscaleColor(const QColor& color)
{
    const int value = qGray(color.rgb());
    return QColor(value, value, value, color.alpha());
}

// Static skin images are converted once, when their cached texture is uploaded.
inline QImage grayscaleImage(const QImage& source)
{
    QImage image = source.convertToFormat(QImage::Format_ARGB32);
    for (int y = 0; y < image.height(); ++y) {
        auto* row = reinterpret_cast<QRgb*>(image.scanLine(y));
        for (int x = 0; x < image.width(); ++x) {
            const int value = qGray(row[x]);
            row[x] = qRgba(value, value, value, qAlpha(row[x]));
        }
    }
    return image;
}

} // namespace miacode
