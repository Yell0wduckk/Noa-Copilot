#pragma once

#include <QImage>

namespace miacode::timeline {

// Keep the skin's highlights, shadows and alpha over the waveform.
inline QRgb candidateNotePixel(QRgb pixel)
{
    const int gray = qGray(pixel);
    const auto tint = [gray](int channel) { return (2 * gray + channel + 1) / 3; };
    return qRgba(tint(qRed(pixel)), tint(qGreen(pixel)), tint(qBlue(pixel)), qAlpha(pixel));
}

inline QColor candidateNoteColor(const QColor& color)
{
    return QColor::fromRgba(candidateNotePixel(color.rgba()));
}

// The texture cache applies this once per skin sprite, including rotated tracks and hold caps.
inline QImage candidateNoteImage(const QImage& source)
{
    QImage result = source.convertToFormat(QImage::Format_ARGB32);
    for (int y = 0; y < result.height(); ++y) {
        auto* row = reinterpret_cast<QRgb*>(result.scanLine(y));
        for (int x = 0; x < result.width(); ++x) {
            row[x] = candidateNotePixel(row[x]);
        }
    }
    return result;
}

} // namespace miacode::timeline
