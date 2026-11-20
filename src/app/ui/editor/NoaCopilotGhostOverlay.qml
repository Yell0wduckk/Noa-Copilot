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

            Text {
                text: root.ghostStreaming
                    ? "续写中..."
                    : (root.ghostOverwriting ? "Tab 采纳替换" : "Tab 采纳")
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: Theme.colors.accent.primary
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        if (!root.noaCopilot) return
                        if (!root.ghostStreaming) root.noaCopilot.acceptGhostAll()
                    }
                }
            }
            Text {
                visible: !root.ghostStreaming
                text: "·"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: Theme.colors.text.disabled
            }
            Text {
                visible: !root.ghostStreaming
                text: "Ctrl+→ 逐拍"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: Theme.colors.text.secondary
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: if (root.noaCopilot) root.noaCopilot.acceptGhostNextStep()
                }
            }
            Text {
                text: "·"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: Theme.colors.text.disabled
            }
            Text {
                text: root.ghostStreaming ? "Esc 停止" : "Esc"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: Theme.colors.text.secondary
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        if (!root.noaCopilot) return
                        if (root.ghostStreaming) root.noaCopilot.cancelActiveGeneration()
                        root.noaCopilot.dismissGhost()
                    }
                }
            }
        }
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
