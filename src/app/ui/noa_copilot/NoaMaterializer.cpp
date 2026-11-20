#include "app/ui/noa_copilot/NoaMaterializer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace miacode::ui {
namespace {
QString number(double value) { return QString::number(value, 'g', 12); }

QString ratio(int ticks, int whole = 384)
{
    const int d = std::gcd(ticks, whole);
    return QStringLiteral("%1:%2").arg(whole / d).arg(ticks / d);
}

QString noteText(const QJsonObject& n, int tick, double bpm)
{
    QString text = n["pad"].toString();
    const int flags = n["flags"].toInt();
    if (flags & 1)
        text += 'b';
    if (flags & 2)
        text += 'x';
    if (flags & 4)
        text += 'f';
    const int kind = n["kind"].toInt();
    if (kind == 1) {
        text += 'h';
        const int duration = n["end"].toInt() - tick;
        if (duration > 0) {
            const double clock = duration * 60.0 / (96 * n["seconds"].toDouble());
            text += '[' + (std::abs(clock - bpm) < 1e-7 ? QString() : number(clock) + '#') + ratio(duration) + ']';
        }
    } else if (kind == 2) {
        const int head = n["head"].toInt();
        if (head == 1)
            text += "$$";
        if (head == 2)
            text += '!';
        QStringList paths;
        for (auto value : n["branches"].toArray()) {
            const auto b = value.toObject();
            QString path;
            for (auto segment : b["path"].toArray())
                path += segment.toString();
            if (b["brk"].toBool())
                path += 'b';
            const int wait = b["launch"].toInt() - tick;
            const int duration = b["end"].toInt() - b["launch"].toInt();
            const double ws = b["wait_seconds"].toDouble(), ds = b["seconds"].toDouble();
            QString timing;
            if (wait && std::abs(ds * wait - ws * duration) < 1e-8) {
                const double tempo = 60 / ws;
                if (std::abs(tempo - bpm) > 1e-7)
                    timing = number(tempo) + '#';
                timing += ratio(duration, 4 * wait);
            } else {
                timing = number(ws) + "##" + number(ds);
            }
            paths << path + '[' + timing + ']';
        }
        text += paths.join('*');
    }
    return text;
}
QString standalone(QString fragment, int division)
{
    if (fragment.isEmpty() || !division)
        return fragment;
    qsizetype i = 0;
    while (i < fragment.size() && fragment[i].isSpace())
        ++i;
    if (i < fragment.size() && fragment[i] == '(')
        i = fragment.indexOf(')', i) + 1;
    if (i < fragment.size() && fragment[i] != '{')
        fragment.insert(i, QStringLiteral("{%1}").arg(division));
    return fragment;
}
}

void NoaMaterializer::begin(const QJsonObject& desc, const QString& original, int beats)
{
    *this = NoaMaterializer();
    layout_ = desc;
    full_ = desc["mode"] == "full";
    skeleton_ = desc["skeleton"].toBool();
    cursor_ = desc["start"].toInt();
    stop_ = desc["stop"].toInt();
    if (full_)
        stop_ = ((stop_ + 95) / 96) * 96;
    origin_ = desc["origin"].toInt();
    width_ = 96 * qMax(1, beats);
    division_ = desc["division"].toInt();
    if (skeleton_)
        source_ = original.mid(desc["left"].toInt(), desc["right"].toInt() - desc["left"].toInt());
    for (auto entry : desc["tempos"].toArray()) {
        const auto row = entry.toArray();
        tempos_.insert(row[0].toInt(), row[1].toDouble());
    }
    for (auto entry : desc["cuts"].toArray()) {
        const auto cut = entry.toObject();
        boundaries_.insert(cut["tick"].toInt(), cut);
    }
}

double NoaMaterializer::bpm(int tick) const
{
    auto it = tempos_.upperBound(tick);
    return it == tempos_.begin() ? 120 : std::prev(it).value();
}

QString NoaMaterializer::interval(int a, int b, int& division, QString body) const
{
    QString out;
    while (a < b) {
        const int boundary = origin_ + ((a - origin_) / width_ + 1) * width_;
        int next = qMin(b, boundary);
        auto change = tempos_.upperBound(a);
        if (change != tempos_.end())
            next = qMin(next, change.key());
        const int size = next - a;
        const int unit = (size % 96 == 0) ? 96 : std::gcd(size, 384);
        if ((a == layout_["start"].toInt() && layout_["initial_bpm"].toDouble() != bpm(a))
            || (tempos_.contains(a) && a != layout_["start"].toInt())) {
            out += '(' + number(bpm(a)) + ')';
        }
        if (division != 384 / unit) {
            division = 384 / unit;
            out += QStringLiteral("{%1}").arg(division);
        }
        out += body + QString(size / unit, ',');
        body.clear();
        if (next == boundary) {
            out += '\n';
            division = 0;
        }
        a = next;
    }
    return out;
}

void NoaMaterializer::advance(int tick)
{
    const int stop = qMin(tick, stop_);
    while (cursor_ < stop) {
        const auto cut = boundaries_.upperBound(cursor_);
        const int next = cut != boundaries_.end() ? qMin(stop, cut.key()) : stop;
        confirmed_ += interval(cursor_, next, division_, pending_);
        pending_.clear();
        cursor_ = next;
        if (boundaries_.contains(next))
            cuts_.append({ int(confirmed_.size()), division_, boundaries_[next] });
    }
}

void NoaMaterializer::moment(int tick, const QJsonArray& notes)
{
    advance(tick);
    int group = -1;
    for (auto value : notes) {
        const auto row = value.toObject();
        const int next = row["group"].toInt();
        if (group >= 0)
            pending_ += next == group ? '/' : QChar(96);
        group = next;
        const QString item = noteText(row, tick, bpm(tick));
        pending_ += item;
        if (row.contains("slot_start"))
            replacements_.insert(row["slot_start"].toInt(), { row["slot_end"].toInt(), item });
        ++noteCount_;
    }
}

QString NoaMaterializer::restore(const QJsonObject& source, double tempo, int division) const
{
    QString tail;
    const double old = source["suffix_bpm"].toDouble();
    if (old > 0 && std::abs(old - tempo) > 1e-7)
        tail += '(' + number(old) + ')';
    const QString div = source["suffix_div"].toString();
    if (!div.isEmpty() && div != QString::number(division))
        tail += '{' + div + '}';
    return tail;
}

void NoaMaterializer::finish()
{
    advance(stop_);
    done_ = true;
}

QString NoaMaterializer::skeletonText(int until) const
{
    const int left = layout_["left"].toInt();
    int pos = left;
    QString result;
    for (auto it = replacements_.cbegin(); it != replacements_.cend() && it.key() < until; ++it) {
        result += source_.mid(pos - left, it.key() - pos) + it->text;
        pos = it->stop;
    }
    return result + source_.mid(pos - left, until - pos);
}

QString NoaMaterializer::rendered(bool preview) const
{
    QString text = confirmed_;
    int div = division_;
    if (!pending_.isEmpty())
        text += interval(cursor_, qMin(stop_, cursor_ + 4), div, pending_);
    if (preview)
        return standalone(text, layout_["division"].toInt());
    if (skeleton_)
        return skeletonText(layout_["right"].toInt());
    if (full_) {
        while (text.endsWith('\n'))
            text.chop(1);
        return text + "\nE";
    }
    if (done_) {
        const int tail = layout_["tail_tick"].toInt(stop_);
        text += interval(stop_, tail, div, {});
        text += restore(layout_, bpm(tail - 1), div);
    }
    return layout_["lead"].toString() + text;
}
QString NoaMaterializer::text() const { return rendered(false); }
QString NoaMaterializer::preview() const { return rendered(true); }

QJsonObject NoaMaterializer::step(int i) const
{
    if (i >= cuts_.size())
        return {};
    const Cut& cut = cuts_[i];
    const int tick = cut.source["tick"].toInt();
    const int pos = cut.source["position"].toInt();
    const int before = i ? cuts_[i - 1].offset : 0;
    QString accepted = confirmed_.mid(before, cut.offset - before);
    if (!i)
        accepted.prepend(layout_["lead"].toString());
    QString rest = confirmed_.mid(cut.offset);
    int div = cut.division;
    QString bridge = interval(tick, cut.source["next_tick"].toInt(), div, {});
    bridge += restore(cut.source, bpm(qMax(tick, cut.source["next_tick"].toInt() - 1)), div);
    QString remainder = text().mid(layout_["lead"].toString().size() + cut.offset);
    if (skeleton_) {
        const QString prefix = skeletonText(pos);
        accepted = i ? prefix.mid(skeletonText(cuts_[i - 1].source["position"].toInt()).size()) : prefix;
        remainder = text().mid(prefix.size());
        bridge.clear();
    }
    return { { "accept_text", accepted }, { "preview_remaining_text", remainder }, { "escape_bridge", bridge },
        { "remaining_ghost_simai", standalone(rest, cut.division) }, { "replace_end_u16", pos },
        { "remaining_start_second", cut.source["second"] }, { "remaining_bpm", bpm(tick) }, { "tick", tick },
        { "division", cut.division } };
}
} // namespace miacode::ui
