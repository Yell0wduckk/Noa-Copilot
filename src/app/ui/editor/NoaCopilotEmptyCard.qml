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
    z: 5
    visible: root.noaCopilot ? root.noaCopilot.emptyCardVisible : false

    property bool advancedExpanded: false

    onVisibleChanged: {
        if (!visible)
            root.advancedExpanded = false
    }

    Item {
        id: cardBox
        anchors.centerIn: parent
        width: Math.min(380, Math.max(260, root.width - 32))
        height: cardCol.implicitHeight + 28

        FloatingCard {
            anchors.fill: parent
        }

        ColumnLayout {
            id: cardCol
            x: 16
            y: 14
            width: parent.width - 32
            spacing: 10

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: Theme.colors.accent.primary
                }

                Text {
                    Layout.fillWidth: true
                    text: "NoaCopilot 整曲智能起稿"
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.headingFontSize
                    font.weight: Font.DemiBold
                    color: Theme.colors.text.active
                }
            }

            Text {
                Layout.fillWidth: true
                text: "当前难度内容为空。可基于音频、BPM 与难度定数直接流式生成整曲初稿，或输入音符后按 Tab 续写。"
                font.family: Theme.uiFont
                font.pixelSize: Theme.secondaryFontSize
                color: Theme.colors.text.secondary
                wrapMode: Text.WordWrap
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

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
                    text: "不再提示"
                    onClicked: {
                        if (root.noaCopilot)
                            root.noaCopilot.dismissEmptyCardForCurrentChart()
                        if (root.editor)
                            root.editor.forceActiveFocus()
                    }
                }

                AppButton {
                    emphasized: true
                    enabled: root.noaCopilot ? !root.noaCopilot.busy : false
                    text: "生成整曲初稿"
                    onClicked: if (root.noaCopilot) root.noaCopilot.triggerFullChartStream()
                }
            }

            Rectangle {
                Layout.fillWidth: true
                visible: root.advancedExpanded
                implicitHeight: emptyAdvCol.implicitHeight + 14
                radius: Theme.compactControlRadius
                color: Theme.surfaceColor(Theme.colors.background.surface)
                border.width: 1 / root.Screen.devicePixelRatio
                border.color: Theme.floatingBorderColor

                ColumnLayout {
                    id: emptyAdvCol
                    x: 8
                    y: 7
                    width: parent.width - 16
                    spacing: 6

                    NoaTemperatureSlider {
                        label: "采样温度"
                        labelWidth: 68
                        labelFontSize: Theme.secondaryFontSize
                        controlEnabled: root.noaCopilot ? root.noaCopilot.enabled : false
                        value: root.noaCopilot ? root.noaCopilot.temperature : 0.5
                        onMoved: v => { if (root.noaCopilot) root.noaCopilot.temperature = v }
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
                                const target = root.noaCopilot.defaultDesigner
                                for (let i = 0; i < opts.length; ++i) {
                                    if (opts[i].value === target) return i
                                }
                                return 0
                            }
                            onActivated: index => {
                                if (root.noaCopilot && index >= 0 && index < root.noaCopilot.designerOptions.length)
                                    root.noaCopilot.defaultDesigner = root.noaCopilot.designerOptions[index].value
                            }
                        }
                    }
                }
            }
        }
    }
}
