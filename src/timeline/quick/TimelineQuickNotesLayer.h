#pragma once

#include "timeline/TimelineSceneState.h"

class QQuickWindow;
class QSGNode;
class TimelineQuickTextureCache;

class TimelineQuickNotesLayer
{
public:
    explicit TimelineQuickNotesLayer(bool grayscale = false) : grayscale_(grayscale) {}
    QSGNode* updateNode(
        QSGNode* oldNode,
        const miacode::timeline::TimelineSceneState& state,
        QQuickWindow* window,
        TimelineQuickTextureCache* textures) const;

private:
    bool grayscale_;
};
