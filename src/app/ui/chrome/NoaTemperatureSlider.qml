import QtQuick
import QtQuick.Layouts
import MiaCode.UI

ColumnLayout {
    id: root

    property string label: "采样温度"
    property int labelWidth: 120
    property int labelFontSize: Theme.uiFontSize
    property real value: 0.5
    property bool controlEnabled: true
    signal moved(real value)

    readonly property int rowSpacing: 8

    Layout.fillWidth: true
    spacing: 2

    RowLayout {
        Layout.fillWidth: true
        spacing: root.rowSpacing

        Text {
            font.family: Theme.uiFont
            font.pixelSize: root.labelFontSize
            Layout.preferredWidth: root.labelWidth
            text: root.label
            color: root.controlEnabled ? Theme.colors.text.secondary : Theme.colors.text.disabled
            wrapMode: Text.WordWrap
        }

        AppSlider {
            id: slider
            Layout.fillWidth: true
            enabled: root.controlEnabled
            from: 0.0
            to: 1.0
            stepSize: 0.05
            Accessible.name: root.label
            Accessible.description: root.value.toFixed(2)
            onMoved: root.moved(value)

            Binding {
                target: slider
                property: "value"
                value: Math.max(0.0, Math.min(1.0, root.value))
                when: !slider.pressed
                restoreMode: Binding.RestoreNone
            }
        }

        EditableValue {
            id: valueEditor
            Layout.preferredWidth: 48
            Layout.preferredHeight: 22
            enabled: root.controlEnabled
            text: Math.max(0.0, Math.min(1.0, root.value)).toFixed(2)
            value: Math.max(0.0, Math.min(1.0, root.value))
            from: 0.0
            to: 1.0
            stepSize: 0.05
            decimals: 2
            onCommitted: function(v) { root.moved(v) }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: root.rowSpacing

        Item {
            Layout.preferredWidth: root.labelWidth
            implicitHeight: 1
        }

        Item {
            Layout.fillWidth: true
            implicitHeight: Math.max(stableHint.implicitHeight, balancedHint.implicitHeight, aggressiveHint.implicitHeight)

            Text {
                id: stableHint
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                text: "稳定"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: !root.controlEnabled
                    ? Theme.colors.text.disabled
                    : (root.value <= 0.25 ? Theme.colors.accent.primary : Theme.colors.text.secondary)

                MouseArea {
                    anchors.fill: parent
                    enabled: root.controlEnabled
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.moved(0.0)
                }
            }

            Text {
                id: balancedHint
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.verticalCenter: parent.verticalCenter
                text: "均衡"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: !root.controlEnabled
                    ? Theme.colors.text.disabled
                    : (root.value > 0.25 && root.value < 0.75 ? Theme.colors.accent.primary : Theme.colors.text.secondary)

                MouseArea {
                    anchors.fill: parent
                    enabled: root.controlEnabled
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.moved(0.5)
                }
            }

            Text {
                id: aggressiveHint
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                text: "激进"
                font.family: Theme.uiFont
                font.pixelSize: Theme.captionFontSize
                color: !root.controlEnabled
                    ? Theme.colors.text.disabled
                    : (root.value >= 0.75 ? Theme.colors.accent.primary : Theme.colors.text.secondary)

                MouseArea {
                    anchors.fill: parent
                    enabled: root.controlEnabled
                    cursorShape: Qt.PointingHandCursor
                    onClicked: root.moved(1.0)
                }
            }
        }

        Item {
            Layout.preferredWidth: 48
            implicitHeight: 1
        }
    }
}
