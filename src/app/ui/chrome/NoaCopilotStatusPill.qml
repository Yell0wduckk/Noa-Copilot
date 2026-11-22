import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Window
import MiaCode.UI

Item {
    id: root

    required property var noaCopilot
    property bool difficultyActive: false
    signal openPreferencesRequested()

    implicitWidth: pillRow.implicitWidth + 12
    implicitHeight: 19

    property bool advancedExpanded: false

    readonly property color dotColor: {
        if (!root.noaCopilot || !root.noaCopilot.enabled)
            return Theme.colors.text.disabled
        const state = root.noaCopilot.daemonState
        if (state === "error")
            return Theme.colors.syntax.error
        if (state === "missing_checkpoint" || state === "warming_audio")
            return Theme.colors.syntax.warning
        if (state === "generating")
            return Theme.colors.accent.primary
        if (state === "ready")
            return "#4CC38A"
        return Theme.colors.text.secondary
    }

    Rectangle {
        anchors.fill: parent
        radius: Theme.smallControlRadius
        color: pillMouse.containsMouse || statusPopup.visible
            ? Theme.overlayColor(Theme.colors.activityState.hover)
            : "transparent"
    }

    RowLayout {
        id: pillRow
        anchors.centerIn: parent
        spacing: 5

        Rectangle {
            width: 6
            height: 6
            radius: 3
            color: root.dotColor
        }

        Text {
            text: root.noaCopilot
                ? ("NoaCopilot · " + (root.noaCopilot.transientHint || root.noaCopilot.statusMessage)
                    + (root.noaCopilot.candidateLocalIssues > 0 ? " · 局部无理 " + root.noaCopilot.candidateLocalIssues : ""))
                : "NoaCopilot"
            font.family: Theme.uiFont
            font.pixelSize: Theme.secondaryFontSize
            color: root.noaCopilot && root.noaCopilot.enabled
                ? Theme.colors.text.status
                : Theme.colors.text.disabled
        }
    }

    MouseArea {
        id: pillMouse
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: {
            if (statusPopup.visible)
                statusPopup.close()
            else {
                if (root.noaCopilot)
                    root.noaCopilot.refreshCheckpoints()
                statusPopup.open()
            }
        }
    }

    FileDialog {
        id: checkpointPicker
        title: "选择 NoaCopilot 模型权重 (.pt / .ckpt)"
        fileMode: FileDialog.OpenFile
        nameFilters: ["Model Checkpoints (*.pt *.ckpt *.safetensors *.bin)", "All Files (*)"]
        onAccepted: {
            if (root.noaCopilot)
                root.noaCopilot.chooseCheckpointFileUrl(selectedFile)
        }
    }

    AppDropdownPanel {
        id: statusPopup
        parent: Overlay.overlay
        width: 340
        height: popupCol.implicitHeight + topPadding + bottomPadding
        x: {
            if (!parent) return 0
            const globalPt = root.mapToItem(parent, root.width - width, 0)
            return Math.max(8, Math.min(globalPt.x, parent.width - width - 8))
        }
        y: {
            if (!parent) return 0
            const globalPt = root.mapToItem(parent, 0, -height - 6)
            return Math.max(8, globalPt.y)
        }
        onClosed: root.advancedExpanded = false

        contentItem: ColumnLayout {
            id: popupCol
            width: statusPopup.availableWidth
            spacing: 8

            // Header + Master Switch
            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: root.dotColor
                }

                Text {
                    Layout.fillWidth: true
                    text: "NoaCopilot 智能制谱助手"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.uiFontSize
                    font.weight: Font.DemiBold
                    color: Theme.colors.text.active
                }

                AppSwitch {
                    checked: root.noaCopilot ? root.noaCopilot.enabled : false
                    onToggled: if (root.noaCopilot) root.noaCopilot.enabled = checked
                }
            }

            // Checkpoint Selector
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 4

                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        Layout.fillWidth: true
                        text: "模型权重"
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.secondaryFontSize
                        color: Theme.colors.text.secondary
                    }
                    ChromeRow {
                        implicitHeight: 20
                        leftPadding: 6
                        rightPadding: 6
                        text: "浏览..."
                        labelFont: Qt.font({ family: Theme.uiFont, pixelSize: Theme.captionFontSize })
                        onClicked: checkpointPicker.open()
                    }
                    ChromeRow {
                        implicitHeight: 20
                        leftPadding: 6
                        rightPadding: 6
                        text: "权重目录"
                        labelFont: Qt.font({ family: Theme.uiFont, pixelSize: Theme.captionFontSize })
                        onClicked: if (root.noaCopilot) root.noaCopilot.openCheckpointDirectory()
                    }
                }

                AppComboBox {
                    Layout.fillWidth: true
                    compact: true
                    textRole: "label"
                    enabled: root.noaCopilot && root.noaCopilot.availableCheckpoints.length > 0
                    model: root.noaCopilot && root.noaCopilot.availableCheckpoints.length > 0
                        ? root.noaCopilot.availableCheckpoints
                        : [{ label: "未检测到模型权重 (点击上方浏览选择)", value: "" }]
                    currentIndex: {
                        if (!root.noaCopilot) return 0
                        const items = root.noaCopilot.availableCheckpoints
                        const cur = root.noaCopilot.checkpointPath
                        for (let i = 0; i < items.length; ++i) {
                            if (items[i].value === cur) return i
                        }
                        return 0
                    }
                    onActivated: index => {
                        if (root.noaCopilot && index >= 0 && index < root.noaCopilot.availableCheckpoints.length)
                            root.noaCopilot.checkpointPath = root.noaCopilot.availableCheckpoints[index].value
                    }
                }
            }

            // Sampling Temperature (0..1)
            NoaTemperatureSlider {
                label: "采样温度"
                labelWidth: 64
                labelFontSize: Theme.secondaryFontSize
                controlEnabled: root.noaCopilot ? root.noaCopilot.enabled : false
                value: root.noaCopilot ? root.noaCopilot.temperature : 0.5
                onMoved: v => { if (root.noaCopilot) root.noaCopilot.temperature = v }
            }

            // Quick Actions when editing a difficulty
            RowLayout {
                Layout.fillWidth: true
                spacing: 6
                visible: root.difficultyActive

                AppButton {
                    Layout.fillWidth: true
                    implicitHeight: Theme.compactControlHeight + 2
                    text: "续写 (Tab)"
                    enabled: root.noaCopilot ? (root.noaCopilot.enabled && !root.noaCopilot.busy) : false
                    onClicked: {
                        statusPopup.close()
                        if (root.noaCopilot) root.noaCopilot.triggerManualContinue()
                    }
                }

                AppButton {
                    Layout.fillWidth: true
                    implicitHeight: Theme.compactControlHeight + 2
                    text: "整曲初稿"
                    enabled: root.noaCopilot ? (root.noaCopilot.enabled && !root.noaCopilot.busy) : false
                    onClicked: {
                        statusPopup.close()
                        if (root.noaCopilot) root.noaCopilot.triggerFullChartStream()
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1 / root.Screen.devicePixelRatio
                color: Theme.separatorColor
            }

            // Collapsible Advanced Settings & Preferences Link
            RowLayout {
                Layout.fillWidth: true
                spacing: 6

                ChromeRow {
                    implicitHeight: 24
                    leftPadding: 6
                    rightPadding: 6
                    text: root.advancedExpanded ? "高级设置 ▴" : "高级设置 ▾"
                    labelFont: Qt.font({ family: Theme.uiFont, pixelSize: Theme.secondaryFontSize })
                    onClicked: root.advancedExpanded = !root.advancedExpanded
                }

                Item { Layout.fillWidth: true }

                ChromeRow {
                    implicitHeight: 24
                    leftPadding: 8
                    rightPadding: 8
                    text: "首选项设置..."
                    labelFont: Qt.font({ family: Theme.uiFont, pixelSize: Theme.secondaryFontSize })
                    onClicked: {
                        statusPopup.close()
                        root.openPreferencesRequested()
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                visible: root.advancedExpanded
                implicitHeight: popupAdvCol.implicitHeight + 14
                radius: Theme.compactControlRadius
                color: Theme.surfaceColor(Theme.colors.background.surface)
                border.width: 1 / root.Screen.devicePixelRatio
                border.color: Theme.floatingBorderColor

                ColumnLayout {
                    id: popupAdvCol
                    x: 8
                    y: 7
                    width: parent.width - 16
                    spacing: 6

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Text {
                            Layout.preferredWidth: 68
                            text: "默认谱师"
                            font.family: Theme.uiFont
                            font.pixelSize: Theme.captionFontSize
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
                                const cur = root.noaCopilot.defaultDesigner
                                for (let i = 0; i < opts.length; ++i) {
                                    if (opts[i].value === cur) return i
                                }
                                return 0
                            }
                            onActivated: index => {
                                if (root.noaCopilot && index >= 0 && index < root.noaCopilot.designerOptions.length)
                                    root.noaCopilot.defaultDesigner = root.noaCopilot.designerOptions[index].value
                            }
                        }
                    }

                    LabeledSlider {
                        label: "slide出现概率"
                        labelWidth: 94
                        from: 1
                        to: 4
                        stepSize: 0.1
                        decimals: 1
                        suffix: "x"
                        value: root.noaCopilot ? root.noaCopilot.slideProbabilityMultiplier : 1
                        onMoved: v => { if (root.noaCopilot) root.noaCopilot.slideProbabilityMultiplier = v }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            Layout.preferredWidth: 68
                            text: "随机种子"
                            color: Theme.colors.text.secondary
                        }
                        AppTextField {
                            Layout.fillWidth: true
                            validator: IntValidator { bottom: -1; top: 2147483647 }
                            text: root.noaCopilot ? String(root.noaCopilot.seed) : "-1"
                            onEditingFinished: {
                                if (acceptableInput && root.noaCopilot)
                                    root.noaCopilot.seed = Number(text)
                                text = Qt.binding(() => root.noaCopilot ? String(root.noaCopilot.seed) : "-1")
                            }
                        }
                        Text {
                            text: "-1 随机"
                            color: Theme.colors.text.secondary
                        }
                    }


                }
            }
        }
    }
}
