#pragma once

#include <QImage>

namespace miacode::timeline {

// Keep the skin's highlights, shadows and alpha over the waveform.
inline QRgb candidateNotePixel(QRgb pixel)
{
    // Halve HSV saturation at the original value. Cyan slide arrows keep their
    // bright edges even where the waveform has a similar luminance.
    const int value = qMax(qRed(pixel), qMax(qGreen(pixel), qBlue(pixel)));
    const auto tint = [value](int channel) { return (value + channel + 1) / 2; };
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
