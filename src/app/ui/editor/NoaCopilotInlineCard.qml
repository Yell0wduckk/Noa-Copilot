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
    z: 6
    visible: root.noaCopilot && (root.noaCopilot.inlineCardVisible || root.noaCopilot.pendingRangeActive)

    property bool advancedExpanded: false

    readonly property rect caretRect: root.editor ? root.editor.cursorRectangle : Qt.rect(24, 24, 2, 18)
    readonly property real cardX: Math.max(12, Math.min(Math.round(root.caretRect.x),
        Math.max(12, root.width - cardContainer.width - 20)))
    readonly property real belowY: Math.round(root.caretRect.y + root.caretRect.height + 6)
    readonly property real aboveY: Math.round(root.caretRect.y - cardContainer.height - 6)
    readonly property real cardY: (belowY + cardContainer.height <= root.height - 8 || aboveY < 8)
        ? Math.max(8, belowY)
        : Math.max(8, aboveY)

    onVisibleChanged: {
        if (!visible)
            root.advancedExpanded = false
    }

    Item {
        id: cardContainer
        x: root.cardX
        y: root.cardY
        width: root.noaCopilot && root.noaCopilot.pendingRangeActive ? 410 : 390
        height: contentCol.implicitHeight + 20

        FloatingCard {
            anchors.fill: parent
        }

        ColumnLayout {
            id: contentCol
            x: 12
            y: 10
            width: parent.width - 24
            spacing: 8

            // Mode A: Ctrl+K Range Generation Setup Card
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8
                visible: root.noaCopilot ? (!root.noaCopilot.pendingRangeActive && root.noaCopilot.inlineCardVisible) : false

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Rectangle {
                        width: 7
                        height: 7
                        radius: 3.5
                        color: Theme.colors.accent.primary
                    }

                    Text {
                        Layout.fillWidth: true
                        text: "NoaCopilot 段落生成"
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.uiFontSize
                        font.weight: Font.DemiBold
                        color: Theme.colors.text.active
                    }

                    Text {
                        text: root.noaCopilot ? root.noaCopilot.statusMessage : ""
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.captionFontSize
                        color: Theme.colors.text.secondary
                    }
                }

                // Span quick choices
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 4

                    Repeater {
                        model: [
                            { beats: 4, label: "4 拍" },
                            { beats: 8, label: "8 拍" },
                            { beats: 16, label: "16 拍" },
                            { beats: 0, label: "填补至下一音符" }
                        ]
                        delegate: AppChoiceButton {
                            required property var modelData
                            Layout.fillWidth: true
                            implicitHeight: Theme.compactControlHeight + 2
                            text: modelData.label
                            checked: root.noaCopilot ? root.noaCopilot.inlineSpanBeats === modelData.beats : false
                            onClicked: if (root.noaCopilot) root.noaCopilot.inlineSpanBeats = modelData.beats
                        }
                    }
                }

                // Collapsible Advanced Settings toggle + Primary Actions
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    ChromeRow {
                        implicitHeight: Theme.compactControlHeight + 2
                        leftPadding: 8
                        rightPadding: 8
                        text: root.advancedExpanded ? "高级设置 ▴" : "高级设置 ▾"
                        labelFont: Qt.font({
                            family: Theme.uiFont,
                            pixelSize: Theme.secondaryFontSize
                        })
                        onClicked: root.advancedExpanded = !root.advancedExpanded
                    }

                    Item { Layout.fillWidth: true }

                    AppButton {
                        implicitHeight: Theme.compactControlHeight + 4
                        text: "取消 (Esc)"
                        onClicked: {
                            if (root.noaCopilot)
                                root.noaCopilot.closeInlineCard()
                            if (root.editor)
                                root.editor.forceActiveFocus()
                        }
                    }

                    AppButton {
                        emphasized: true
                        implicitHeight: Theme.compactControlHeight + 4
                        enabled: root.noaCopilot ? !root.noaCopilot.busy : false
                        text: root.noaCopilot && root.noaCopilot.busy ? "生成中..." : "生成 (Enter)"
                        onClicked: if (root.noaCopilot) root.noaCopilot.triggerRangeGenerate()
                    }
                }

                // Collapsible Advanced Settings content
                Rectangle {
                    Layout.fillWidth: true
                    visible: root.advancedExpanded
                    implicitHeight: advCol.implicitHeight + 14
                    radius: Theme.compactControlRadius
                    color: Theme.surfaceColor(Theme.colors.background.surface)
                    border.width: 1 / root.Screen.devicePixelRatio
                    border.color: Theme.floatingBorderColor

                    ColumnLayout {
                        id: advCol
                        x: 8
                        y: 7
                        width: parent.width - 16
                        spacing: 6

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Text {
                                Layout.preferredWidth: 68
                                text: "目标定数"
                                font.family: Theme.uiFont
                                font.pixelSize: Theme.secondaryFontSize
                                color: Theme.colors.text.secondary
                            }

                            AppTextField {
                                Layout.fillWidth: true
                                implicitHeight: Theme.compactControlHeight + 2
                                placeholderText: "留空则跟随当前难度定数"
                                text: root.noaCopilot ? root.noaCopilot.inlineLevelOverride : ""
                                onTextEdited: if (root.noaCopilot) root.noaCopilot.inlineLevelOverride = text
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Text {
                                Layout.preferredWidth: 68
                                text: "谱师风格"
                                font.family: Theme.uiFont
                                font.pixelSize: Theme.secondaryFontSize
                                color: Theme.colors.text.secondary
                            }

                            AppComboBox {
                                Layout.fillWidth: true
                                compact: true
                                textRole: "label"
                                model: root.noaCopilot ? root.noaCopilot.designerOptions : []
                                currentIndex: {
                                    if (!root.noaCopilot) return 0
                                    const opts = root.noaCopilot.designerOptions
                                    const target = root.noaCopilot.inlineDesignerOverride
                                    for (let i = 0; i < opts.length; ++i) {
                                        if (opts[i].value === target) return i
                                    }
                                    return 0
                                }
                                onActivated: index => {
                                    if (root.noaCopilot && index >= 0 && index < root.noaCopilot.designerOptions.length)
                                        root.noaCopilot.inlineDesignerOverride = root.noaCopilot.designerOptions[index].value
                                }
                            }
                        }
                    }
                }
            }

            // Mode B: Pending Range Diff Review & Audition Bar
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8
                visible: root.noaCopilot ? root.noaCopilot.pendingRangeActive : false

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Rectangle {
                        width: 7
                        height: 7
                        radius: 3.5
                        color: Theme.colors.accent.primary
                    }

                    Text {
                        Layout.fillWidth: true
                        text: root.noaCopilot
                            ? ("候选段落已就绪 · " + root.noaCopilot.pendingRangeNoteCount + " 音符")
                            : ""
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.uiFontSize
                        font.weight: Font.DemiBold
                        color: Theme.colors.text.active
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    AppButton {
                        emphasized: true
                        Layout.fillWidth: true
                        implicitHeight: Theme.compactControlHeight + 4
                        text: "接受 (Tab)"
                        onClicked: {
                            if (root.noaCopilot)
                                root.noaCopilot.acceptPendingRange()
                            if (root.editor)
                                root.editor.forceActiveFocus()
                        }
                    }

                    AppButton {
                        Layout.fillWidth: true
                        implicitHeight: Theme.compactControlHeight + 4
                        enabled: root.noaCopilot ? !root.noaCopilot.busy : false
                        text: "重抽 (R)"
                        onClicked: if (root.noaCopilot) root.noaCopilot.regeneratePendingRange()
                    }

                    AppButton {
                        Layout.fillWidth: true
                        implicitHeight: Theme.compactControlHeight + 4
                        text: "试听 (Space)"
                        onClicked: if (root.noaCopilot) root.noaCopilot.auditionPendingRange()
                    }

                    AppButton {
                        Layout.fillWidth: true
                        implicitHeight: Theme.compactControlHeight + 4
                        text: "放弃 (Esc)"
                        onClicked: {
                            if (root.noaCopilot)
                                root.noaCopilot.rejectPendingRange()
                            if (root.editor)
                                root.editor.forceActiveFocus()
                        }
                    }
                }
            }
        }
    }
}
