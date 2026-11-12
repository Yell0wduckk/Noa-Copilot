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
    readonly property rect caretRect: root.editor ? root.editor.cursorRectangle : Qt.rect(0, 0, 2, 18)

    // Inline first-line ghost text right after the Scintilla caret
    Row {
        id: inlineGhostRow
        visible: root.ghostVisible
        x: Math.round(root.caretRect.x + 2)
        y: Math.round(root.caretRect.y)
        height: Math.max(18, root.caretRect.height)
        spacing: 8

        Text {
            id: firstLineText
            anchors.verticalCenter: parent.verticalCenter
            text: root.noaCopilot ? root.noaCopilot.ghostFirstLine : ""
            font: root.editor ? root.editor.effectiveFont : Theme.codeFont
            color: Theme.colors.text.editor
            opacity: 0.45
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
            opacity: 0.9

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

    // Extra multi-line ghost preview card below the current line
    Rectangle {
        id: extraLinesCard
        visible: root.ghostVisible && root.noaCopilot && root.noaCopilot.ghostExtraLines.length > 0
        x: Math.max(8, Math.min(Math.round(root.caretRect.x), Math.max(8, root.width - width - 16)))
        y: Math.round(root.caretRect.y + root.caretRect.height + 2)
        width: Math.min(Math.max(160, extraText.implicitWidth + 16), Math.max(160, root.width - 24))
        height: extraText.implicitHeight + 10
        radius: Theme.compactControlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        border.width: 1 / root.Screen.devicePixelRatio
        border.color: Theme.floatingBorderColor
        opacity: 0.92

        Text {
            id: extraText
            x: 8
            y: 5
            width: parent.width - 16
            text: root.noaCopilot ? root.noaCopilot.ghostExtraLines : ""
            font: root.editor ? root.editor.effectiveFont : Theme.codeFont
            color: Theme.colors.text.editor
            opacity: 0.55
            wrapMode: Text.WrapAnywhere
        }
    }

    // Bottom-right non-blocking streaming indicator when full chart generation is active
    Rectangle {
        id: fullStreamBanner
        visible: root.noaCopilot ? root.noaCopilot.fullStreamingActive : false
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: 18
        anchors.bottomMargin: 14
        width: streamRow.implicitWidth + 20
        height: 32
        radius: Theme.controlRadius
        color: Theme.surfaceColor(Theme.colors.background.elevated)
        border.width: 1 / root.Screen.devicePixelRatio
        border.color: Theme.colors.accent.primary

        RowLayout {
            id: streamRow
            anchors.centerIn: parent
            spacing: 10

            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: Theme.colors.accent.primary
            }

            Text {
                text: root.noaCopilot
                    ? ("NoaCopilot 正在流式生成整曲 ("
                       + root.noaCopilot.fullStreamingBeat + "/"
                       + Math.max(1, root.noaCopilot.fullStreamingTotalBeats) + " 拍)")
                    : ""
                font.family: Theme.uiFont
                font.pixelSize: Theme.secondaryFontSize
                color: Theme.colors.text.primary
            }

            ChromeRow {
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
