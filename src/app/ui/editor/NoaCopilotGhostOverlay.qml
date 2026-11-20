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
    readonly property rect caretRect: root.editor ? root.editor.cursorRectangle : Qt.rect(0, 0, 2, 18)

    readonly property rect strikeStartRect: {
        if (!root.ghostOverwriting || !root.editor)
            return root.caretRect
        const _v = root.editor.vertical_scroll_value
        const _h = root.editor.horizontal_scroll_value
        const _c = root.caretRect
        const src = root.editor.text || ""
        let pos = Math.max(0, Math.min(root.noaCopilot.ghostReplaceStart, src.length))
        const endPos = Math.max(pos, Math.min(root.noaCopilot.ghostReplaceEnd, src.length))
        while (pos < endPos && (src[pos] === "\n" || src[pos] === "\r"))
            pos += 1
        return root.editor.textPositionRectangle(pos)
    }

    readonly property rect strikeEndRect: {
        if (!root.ghostOverwriting || !root.editor)
            return root.caretRect
        const _v = root.editor.vertical_scroll_value
        const _h = root.editor.horizontal_scroll_value
        const _c = root.caretRect
        const src = root.editor.text || ""
        const startPos = Math.max(0, Math.min(root.noaCopilot.ghostReplaceStart, src.length))
        let pos = Math.max(startPos, Math.min(root.noaCopilot.ghostReplaceEnd - 1, src.length))
        while (pos > startPos && (src[pos] === "\n" || src[pos] === "\r"))
            pos -= 1
        return root.editor.textPositionRectangle(pos)
    }

    // Inline first-line ghost text right after the Scintilla caret (pure append mode)
    Row {
        id: inlineGhostRow
        visible: root.ghostVisible && !root.ghostOverwriting
        x: Math.round(root.caretRect.x + 2)
        y: Math.round(root.caretRect.y)
        height: Math.max(18, root.caretRect.height)
        spacing: 8

        Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            width: firstLineText.implicitWidth + 8
            height: Math.max(18, firstLineText.implicitHeight + 2)
            radius: Theme.smallControlRadius
            color: Theme.contentOverlayColor(Theme.colors.text.secondary, 0.16)

            Text {
                id: firstLineText
                anchors.centerIn: parent
                text: root.noaCopilot ? root.noaCopilot.ghostFirstLine : ""
                font: root.editor ? root.editor.effectiveFont : Theme.codeFont
                color: Theme.colors.text.editor
                opacity: 0.62
            }
        }

        Rectangle {
            id: actionHintBadge
            anchors.verticalCenter: parent.verticalCenter
            height: Math.max(18, hintRow.implicitHeight + 4)
            width: hintRow.implicitWidth + 12
            radius: Theme.smallControlRadius
            color: Theme.surfaceColor(Theme.colors.background.elevated)
            border.width: 1 / root.Screen.devicePixelRatio
            border.color: Theme.floatingBorderColor
            opacity: 0.92

            Row {
                id: hintRow
                anchors.centerIn: parent
                spacing: 6

                Text {
                    text: "Tab 采纳"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: Theme.colors.accent.primary
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: if (root.noaCopilot) root.noaCopilot.acceptGhostAll()
                    }
                }
                Text {
                    text: "·"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: Theme.colors.text.disabled
                }
                Text {
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
                    text: "Esc"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: Theme.colors.text.secondary
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: if (root.noaCopilot) root.noaCopilot.dismissGhost()
                    }
                }
            }
        }
    }

    // Extra multi-line ghost preview card below the current line (pure append mode)
    Rectangle {
        id: extraLinesCard
        visible: root.ghostVisible && !root.ghostOverwriting
            && root.noaCopilot && root.noaCopilot.ghostExtraLines.length > 0
        x: Math.max(8, Math.min(Math.round(root.caretRect.x), Math.max(8, root.width - width - 16)))
        y: Math.round(root.caretRect.y + root.caretRect.height + 2)
        width: Math.min(Math.max(160, extraText.implicitWidth + 16), Math.max(160, root.width - 24))
        height: extraText.implicitHeight + 10
        radius: Theme.compactControlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        border.width: 1 / root.Screen.devicePixelRatio
        border.color: Theme.floatingBorderColor
        opacity: 0.94

        Rectangle {
            anchors.fill: parent
            radius: parent.radius
            color: Theme.contentOverlayColor(Theme.colors.text.secondary, 0.14)
        }

        Text {
            id: extraText
            x: 8
            y: 5
            width: parent.width - 16
            text: root.noaCopilot ? root.noaCopilot.ghostExtraLines : ""
            font: root.editor ? root.editor.effectiveFont : Theme.codeFont
            color: Theme.colors.text.editor
            opacity: 0.65
            wrapMode: Text.WrapAnywhere
        }
    }

    // Gray-backed replacement ghost card below struck-through chart text (mid-paragraph overwrite mode)
    Rectangle {
        id: overwriteGhostCard
        visible: root.ghostOverwriting
        readonly property real belowY: Math.round(
            Math.max(root.caretRect.y + root.caretRect.height,
                     root.strikeEndRect.y + root.strikeEndRect.height) + 4)
        readonly property real aboveY: Math.round(
            Math.min(root.caretRect.y, root.strikeStartRect.y) - height - 4)
        x: Math.max(8, Math.min(Math.round(root.strikeStartRect.x), Math.max(8, root.width - width - 16)))
        y: (belowY + height <= root.height - 8 || aboveY < 8)
            ? Math.max(8, belowY)
            : Math.max(8, aboveY)
        width: Math.min(
            Math.max(210, Math.max(overwriteGhostText.implicitWidth + 24, overwriteHintRow.implicitWidth + 24)),
            Math.max(210, root.width - 24))
        height: overwriteCol.implicitHeight + 14
        radius: Theme.compactControlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        border.width: 1 / root.Screen.devicePixelRatio
        border.color: Theme.floatingBorderColor
        opacity: 0.96

        ColumnLayout {
            id: overwriteCol
            x: 8
            y: 7
            width: parent.width - 16
            spacing: 6

            Rectangle {
                Layout.fillWidth: true
                implicitHeight: overwriteGhostText.implicitHeight + 10
                radius: Theme.smallControlRadius
                color: Theme.contentOverlayColor(Theme.colors.text.secondary, 0.16)

                Text {
                    id: overwriteGhostText
                    x: 8
                    y: 5
                    width: parent.width - 16
                    text: root.noaCopilot ? root.noaCopilot.ghostText : ""
                    font: root.editor ? root.editor.effectiveFont : Theme.codeFont
                    color: Theme.colors.text.editor
                    opacity: 0.72
                    wrapMode: Text.WrapAnywhere
                }
            }

            Row {
                id: overwriteHintRow
                spacing: 6

                Text {
                    text: "Tab 采纳替换"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: Theme.colors.accent.primary
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: if (root.noaCopilot) root.noaCopilot.acceptGhostAll()
                    }
                }
                Text {
                    text: "·"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: Theme.colors.text.disabled
                }
                Text {
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
                    text: "Esc 取消"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.captionFontSize
                    color: Theme.colors.text.secondary
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: if (root.noaCopilot) root.noaCopilot.dismissGhost()
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
