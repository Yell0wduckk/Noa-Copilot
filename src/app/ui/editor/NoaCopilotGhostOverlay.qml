import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import MiaCode.UI

Item {
    id: root

    required property var editor
    required property var noaCopilot

    anchors.fill: parent
    z: 4
    clip: true

    readonly property bool ghostVisible: root.noaCopilot
        && root.noaCopilot.ghostActive
        && !root.noaCopilot.inlineCardVisible
        && !root.noaCopilot.pendingRangeActive
        && root.editor
        && !root.editor.readonly
    readonly property bool ghostOverwriting: root.ghostVisible
        && root.noaCopilot.ghostReplaceEnd > root.noaCopilot.ghostReplaceStart
    readonly property bool ghostStreaming: root.ghostVisible
        && root.noaCopilot.ghostStreaming
    readonly property rect caretRect: root.editor ? root.editor.cursorRectangle : Qt.rect(0, 0, 2, 18)
    readonly property rect ghostStartRect: root.editor && root.editor.inlineGhostActive
        ? root.editor.ghostStartRectangle
        : root.caretRect
    readonly property rect ghostEndRect: root.editor && root.editor.inlineGhostActive
        ? root.editor.ghostEndRectangle
        : root.caretRect

    // Compact floating action hint pill positioned next to / below the inline Scintilla ghost text
    Rectangle {
        id: actionHintBadge
        visible: root.ghostVisible
        readonly property bool singleLineGhost: Math.abs(root.ghostEndRect.y - root.ghostStartRect.y) < 4
        readonly property real inlineX: Math.round(root.ghostEndRect.x + 10)
        readonly property bool fitsInlineRight: singleLineGhost && (inlineX + width <= root.width - 12)
        readonly property real belowY: Math.round(
            Math.max(root.caretRect.y + root.caretRect.height,
                     root.ghostEndRect.y + root.ghostEndRect.height) + 4)
        readonly property real aboveY: Math.round(
            Math.min(root.caretRect.y, root.ghostStartRect.y) - height - 4)

        x: fitsInlineRight
            ? inlineX
            : Math.max(8, Math.min(Math.round(root.ghostStartRect.x), Math.max(8, root.width - width - 12)))
        y: fitsInlineRight
            ? Math.max(4, Math.round(root.ghostEndRect.y + (root.ghostEndRect.height - height) * 0.5))
            : ((belowY + height <= root.height - 8 || aboveY < 8)
                ? Math.max(8, belowY)
                : Math.max(8, aboveY))
        height: Math.max(22, hintRow.implicitHeight + 6)
        width: hintRow.implicitWidth + 14
        radius: Theme.smallControlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        border.width: 1 / root.Screen.devicePixelRatio
        border.color: Theme.floatingBorderColor
        opacity: 0.94

        Row {
            id: hintRow
            anchors.centerIn: parent
            spacing: 6

            Rectangle {
                visible: root.ghostStreaming
                anchors.verticalCenter: parent.verticalCenter
                width: 6
                height: 6
                radius: 3
                color: Theme.colors.accent.primary
            }

            Repeater {
                model: root.noaCopilot ? [root.noaCopilot.candidateHint, "↑ 原谱", "↓ 换一个", "→ 一拍", "Tab 采纳", "Esc 还原"] : []
                delegate: Text {
                    required property int index
                    required property string modelData
                    text: modelData
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: index === 0 ? Theme.colors.accent.primary : Theme.colors.text.secondary
                    MouseArea {
                        anchors.fill: parent
                        enabled: index > 0
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            if (index === 1 || index === 2) root.noaCopilot.switchCandidate(index === 1 ? -1 : 1)
                            if (index === 3) root.noaCopilot.acceptNextGhostStep()
                            if (index === 4) root.noaCopilot.acceptGhostAll()
                            if (index === 5) root.noaCopilot.dismissAllActivePreviews()
                        }
                    }
                }
            }
        }
    }

    Rectangle {
        visible: root.ghostVisible && root.noaCopilot.selectedCandidate > 0
        x: 1
        y: Math.max(0, root.ghostStartRect.y)
        width: 2
        height: Math.max(root.ghostStartRect.height, root.ghostEndRect.y + root.ghostEndRect.height - y)
        color: Theme.colors.accent.primary
        opacity: 0.6
    }
    Rectangle {
        visible: !!root.noaCopilot && root.noaCopilot.enabled && root.noaCopilot.skeletonSelection && !root.ghostVisible
        x: Math.max(4, Math.min(root.width-width-4,root.caretRect.x))
        y: Math.max(4, Math.min(root.height-height-4,root.caretRect.y+root.caretRect.height))
        width: fillHint.implicitWidth+16
        height: 24
        radius: Theme.smallControlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        Text { id: fillHint; anchors.centerIn: parent; text: "填充采音 (Tab)"; color: Theme.colors.accent.primary; font.pixelSize: Theme.captionFontSize }
        MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: root.noaCopilot.triggerManualContinue() }
    }
    // Bottom-right non-blocking indicator for audio analysis & full chart streaming
    Rectangle {
        id: activityBanner
        readonly property bool analyzing: root.noaCopilot ? root.noaCopilot.audioAnalyzing : false
        readonly property bool streaming: root.noaCopilot ? root.noaCopilot.fullStreamingActive : false
        visible: analyzing || streaming
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: 18
        anchors.bottomMargin: 14
        width: streamRow.implicitWidth + 20
        height: 32
        radius: Theme.controlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        border.width: 1 / root.Screen.devicePixelRatio
        border.color: analyzing ? Theme.colors.syntax.warning : Theme.colors.accent.primary

        RowLayout {
            id: streamRow
            anchors.centerIn: parent
            spacing: 10

            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: activityBanner.analyzing ? Theme.colors.syntax.warning : Theme.colors.accent.primary
            }

            Text {
                text: {
                    if (!root.noaCopilot)
                        return ""
                    if (activityBanner.analyzing)
                        return "NoaCopilot 正在分析音频..."
                    if (root.noaCopilot.fullStreamingTotalBeats > 0) {
                        return "NoaCopilot 正在流式生成整曲 ("
                            + root.noaCopilot.fullStreamingBeat + "/"
                            + root.noaCopilot.fullStreamingTotalBeats + " 拍)"
                    }
                    return "NoaCopilot 正在流式生成整曲..."
                }
                font.family: Theme.uiFont
                font.pixelSize: Theme.secondaryFontSize
                color: Theme.colors.text.primary
            }

            ChromeRow {
                visible: activityBanner.streaming
                implicitHeight: 22
                leftPadding: 8
                rightPadding: 8
                text: "停止 (Esc)"
                labelFont: Qt.font({
                    family: Theme.uiFont,
                    pixelSize: Theme.captionFontSize,
                    weight: Font.DemiBold
                })
                onClicked: if (root.noaCopilot) root.noaCopilot.cancelActiveGeneration()
            }
        }
    }
}
