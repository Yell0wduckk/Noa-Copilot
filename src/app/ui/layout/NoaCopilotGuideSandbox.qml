pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import MiaCode.UI

Popup {
    id: root

    // 0: wait Tab (start)
    // 1: streaming Group 1
    // 2: wait Down (1 -> 2)
    // 3: wait Down (2 -> 3)
    // 4: wait Up   (3 -> 2)
    // 5: wait Right (accept beat 1)
    // 6: wait Right (accept beat 2)
    // 7: wait Tab  (accept rest)
    // 8: finished
    property int stage: 0
    property int candidateIndex: 0
    property int streamBeat: 0
    property int acceptedBeats: 0
    property bool acceptedAll: false
    property string toastMessage: ""

    readonly property Item guideTargetItem: sandboxCard
    readonly property bool completed: stage === 8
    readonly property string expectedKey: {
        if (stage === 0 || stage === 7)
            return "Tab"
        if (stage === 2 || stage === 3)
            return "↓"
        if (stage === 4)
            return "↑"
        if (stage === 5 || stage === 6)
            return "→"
        return ""
    }
    readonly property int completedSteps: {
        if (stage === 0)
            return 0
        if (stage === 1 || stage === 2)
            return 1
        if (stage === 3)
            return 2
        if (stage === 4)
            return 3
        if (stage === 5)
            return 4
        if (stage === 6)
            return 5
        if (stage === 7)
            return 6
        return 7
    }
    readonly property string statusText: {
        if (stage === 0)
            return qsTrId("guide.noa.sandbox.status.ready")
        if (stage === 1)
            return qsTrId("guide.noa.sandbox.status.streaming")
        if (stage === 8)
            return qsTrId("guide.noa.sandbox.status.accepted")
        if (acceptedBeats > 0)
            return qsTrId("guide.noa.sandbox.status.stepped").arg(candidateIndex).arg(acceptedBeats)
        return qsTrId("guide.noa.sandbox.status.candidate").arg(candidateIndex)
    }
    readonly property string instructionText: {
        switch (stage) {
        case 0: return qsTrId("guide.noa.sandbox.prompt.tab_start")
        case 1: return qsTrId("guide.noa.sandbox.prompt.streaming")
        case 2: return qsTrId("guide.noa.sandbox.prompt.down_1")
        case 3: return qsTrId("guide.noa.sandbox.prompt.down_2")
        case 4: return qsTrId("guide.noa.sandbox.prompt.up")
        case 5: return qsTrId("guide.noa.sandbox.prompt.right_1")
        case 6: return qsTrId("guide.noa.sandbox.prompt.right_2")
        case 7: return qsTrId("guide.noa.sandbox.prompt.tab_accept")
        default: return qsTrId("guide.noa.sandbox.prompt.done")
        }
    }

    readonly property var prefixLines: [
        "(200){16}2,3,4,5,7,5,6,4,8,7,6,5,2,4,1,5,",
        "{48}2,,,3,,,4,,,5,,,68,,,5,,,4,,,3,,,8/2,,7,,6,,5,,4,,3,,2,,1,,8,,7,,6,,5,,"
    ]
    readonly property var group1Beats: [
        "{4}4/7>8-5p4[4:3],", ",", ",", ",",
        "\n{4},", ",", ",", ",",
        "\n{4},", ",", ",", ",",
        "\n{4},", ",", ",", ","
    ]
    readonly property var group2Lines: [
        "{12}24,3,3,64,5,5,38,2,1,68,7,8,",
        "{32}1b/4b,,,,,,2b/7b,,,,,,8b/5b,,,,3,4,3,4,3,4,3,4,25,,,,,,,,",
        "{12}7,5,6,3,5,4,6,5,5,4,5,5,",
        "{6}6b-8[16:1],2b-4[16:1],1b-5[16:1],3b-8[16:1],7b-4[16:1],1b,"
    ]
    readonly property string group2Step1Committed: "{12}24,3,3,"
    readonly property string group2Step2Committed: "{12}24,3,3,64,5,5,"
    readonly property var group3Lines: [
        "{32}4,6,4,6,4,6,4,6,3,7,3,7,2,8,2,8,1,1,1,1,1,,,,B1/B2/E2f/A2/D2/E3,,,,,,,,",
        "{32}5,3,5,3,5,3,5,3,6,2,6,2,7,1,7,1,8,8,8,8,8,,,,B7/E7/A7/B8/E8f/D8,,,,,,,,",
        "{32}1,2,3,4,5,6,7,8,1,2,3,4,1,2,3,4,8,7,6,5,4,3,2,1,8,7,6,5,8,7,6,5,",
        "{32}2,3,4,5,2,3,4,5,6pp4[4:1]*pp4[4:1],7,8,1,2,3,4,5,,,,,,,,,,,,,,,,,"
    ]
    readonly property var stepSequence: ["Tab", "↓", "↓", "↑", "→", "→", "Tab"]
    readonly property var displayRows: buildRows()

    parent: Overlay.overlay
    popupType: Popup.Item
    modal: false
    dim: false
    focus: false
    closePolicy: Popup.NoAutoClose
    z: 99
    padding: 0
    margins: 0
    enter: null
    exit: null
    background: Item {}

    width: Math.min(780, Math.max(360, (parent ? parent.width : 800) - 2 * Theme.dialogMargin))
    height: Math.min(sandboxColumn.implicitHeight + 2 * Theme.dialogPadding,
        Math.max(220, (parent ? parent.height : 600) - 2 * Theme.dialogMargin))
    x: parent ? Math.round((parent.width - width) / 2) : 0
    y: {
        if (!parent)
            return 0
        const desiredBottomSpace = 270
        const centered = Math.round((parent.height - height) / 2)
        const lifted = Math.max(Theme.dialogMargin, parent.height - height - desiredBottomSpace)
        return Math.max(Theme.dialogMargin, Math.min(centered, lifted))
    }

    onAboutToShow: resetDemo()
    onClosed: {
        streamTimer.stop()
        toastTimer.stop()
    }

    function resetDemo() {
        streamTimer.stop()
        toastTimer.stop()
        stage = 0
        candidateIndex = 0
        streamBeat = 0
        acceptedBeats = 0
        acceptedAll = false
        toastMessage = ""
    }

    function showToast(text) {
        toastMessage = text
        toastTimer.restart()
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
    }

    function colorHex(c) {
        const q = Qt.color(c)
        const hex = v => Math.max(0, Math.min(255, Math.round(v * 255))).toString(16).padStart(2, "0")
        return "#" + hex(q.r) + hex(q.g) + hex(q.b)
    }

    function formatCommitted(text) {
        if (!text)
            return ""
        const escaped = escapeHtml(text)
        const kw = colorHex(Theme.colors.syntax.keyword)
        const dur = colorHex(Theme.colors.syntax.duration)
        return escaped
            .replace(/(\([^)]+\)|\{[^}]+\})/g, "<span style=\"color:" + kw + ";\">$1</span>")
            .replace(/(\[[^\]]+\])/g, "<span style=\"color:" + dur + ";\">$1</span>")
    }

    function buildRows() {
        const rows = [
            { lineNumber: 1, committed: prefixLines[0], ghost: "", caret: false, accentBar: false },
            { lineNumber: 2, committed: prefixLines[1], ghost: "", caret: false, accentBar: false }
        ]

        if (stage === 0) {
            rows.push({ lineNumber: 3, committed: "", ghost: "", caret: true, accentBar: false })
            rows.push({ lineNumber: 4, committed: "", ghost: "", caret: false, accentBar: false })
            rows.push({ lineNumber: 5, committed: "", ghost: "", caret: false, accentBar: false })
            rows.push({ lineNumber: 6, committed: "", ghost: "", caret: false, accentBar: false })
            return rows
        }

        if (candidateIndex === 1) {
            const joined = group1Beats.slice(0, streamBeat).join("")
            const lines = joined.length > 0 ? joined.split("\n") : [""]
            for (let i = 0; i < 4; ++i) {
                const lineText = i < lines.length ? lines[i] : ""
                rows.push({
                    lineNumber: 3 + i,
                    committed: "",
                    ghost: lineText,
                    caret: i === 0,
                    accentBar: false
                })
            }
            return rows
        }

        if (candidateIndex === 2) {
            if (acceptedAll) {
                for (let i = 0; i < group2Lines.length; ++i) {
                    rows.push({
                        lineNumber: 3 + i,
                        committed: group2Lines[i],
                        ghost: "",
                        caret: i === group2Lines.length - 1,
                        accentBar: false
                    })
                }
                return rows
            }
            const firstCommitted = acceptedBeats === 1
                ? group2Step1Committed
                : (acceptedBeats >= 2 ? group2Step2Committed : "")
            const firstGhost = group2Lines[0].slice(firstCommitted.length)
            rows.push({
                lineNumber: 3,
                committed: firstCommitted,
                ghost: firstGhost,
                caret: true,
                accentBar: true
            })
            for (let i = 1; i < group2Lines.length; ++i) {
                rows.push({
                    lineNumber: 3 + i,
                    committed: "",
                    ghost: group2Lines[i],
                    caret: false,
                    accentBar: true
                })
            }
            return rows
        }

        // Candidate 3 (Xaleid◆scopiX Re:MASTER 4 lines at BPM 200)
        for (let i = 0; i < group3Lines.length; ++i) {
            rows.push({
                lineNumber: 3 + i,
                committed: "",
                ghost: group3Lines[i],
                caret: i === 0,
                accentBar: true
            })
        }
        return rows
    }

    function handleKey(key) {
        if (key === Qt.Key_Tab) {
            if (stage === 0) {
                candidateIndex = 1
                streamBeat = 1
                stage = 1
                toastMessage = ""
                streamTimer.start()
                return true
            }
            if (stage === 7) {
                acceptedAll = true
                stage = 8
                toastMessage = ""
                return true
            }
            if (stage === 1) {
                showToast(qsTrId("guide.noa.sandbox.toast.wait_stream"))
                return true
            }
            if (stage < 8) {
                showToast(qsTrId("guide.noa.sandbox.toast.expected").arg(expectedKey))
                return true
            }
            return false
        }

        if (key === Qt.Key_Down) {
            if (stage === 1) {
                showToast(qsTrId("guide.noa.sandbox.toast.wait_stream"))
                return true
            }
            if (stage === 2) {
                candidateIndex = 2
                stage = 3
                toastMessage = ""
                return true
            }
            if (stage === 3) {
                candidateIndex = 3
                stage = 4
                toastMessage = ""
                return true
            }
            if (stage < 8) {
                showToast(qsTrId("guide.noa.sandbox.toast.expected").arg(expectedKey))
                return true
            }
            return false
        }

        if (key === Qt.Key_Up) {
            if (stage === 4) {
                candidateIndex = 2
                stage = 5
                toastMessage = ""
                return true
            }
            if (stage === 1) {
                showToast(qsTrId("guide.noa.sandbox.toast.wait_stream"))
                return true
            }
            if (stage < 8) {
                showToast(qsTrId("guide.noa.sandbox.toast.expected").arg(expectedKey))
                return true
            }
            return false
        }

        if (key === Qt.Key_Right) {
            if (stage === 5) {
                acceptedBeats = 1
                stage = 6
                toastMessage = ""
                return true
            }
            if (stage === 6) {
                acceptedBeats = 2
                stage = 7
                toastMessage = ""
                return true
            }
            if (stage === 1) {
                showToast(qsTrId("guide.noa.sandbox.toast.wait_stream"))
                return true
            }
            if (stage < 8) {
                showToast(qsTrId("guide.noa.sandbox.toast.expected").arg(expectedKey))
                return true
            }
            return false
        }

        return false
    }

    function triggerSymbol(symbol) {
        if (symbol === "Tab")
            return handleKey(Qt.Key_Tab)
        if (symbol === "↓")
            return handleKey(Qt.Key_Down)
        if (symbol === "↑")
            return handleKey(Qt.Key_Up)
        if (symbol === "→")
            return handleKey(Qt.Key_Right)
        return false
    }

    Timer {
        id: streamTimer
        interval: 75
        repeat: true
        onTriggered: {
            if (root.streamBeat < root.group1Beats.length) {
                root.streamBeat += 1
            } else {
                streamTimer.stop()
                if (root.stage === 1) {
                    root.stage = 2
                    root.toastMessage = ""
                }
            }
        }
    }

    Timer {
        id: toastTimer
        interval: 1600
        onTriggered: root.toastMessage = ""
    }

    Timer {
        id: caretBlink
        interval: 530
        repeat: true
        running: root.visible
        property bool on: true
        onTriggered: on = !on
    }

    contentItem: Item {
        id: sandboxCard

        FloatingCard {
            anchors.fill: parent
            cornerRadius: Theme.popupRadius
            tintColor: Theme.colors.background.elevated
            shadowOpacity: Theme.dialogShadowOpacity
        }

        ColumnLayout {
            id: sandboxColumn
            anchors.fill: parent
            anchors.margins: Theme.dialogPadding
            spacing: Theme.panelPadding

            // Header bar
            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Rectangle {
                    Layout.preferredWidth: 8
                    Layout.preferredHeight: 8
                    radius: 4
                    color: root.stage === 1
                        ? Theme.colors.accent.primary
                        : (root.completed ? "#4CC38A" : Theme.colors.accent.primary)
                }

                Text {
                    Layout.fillWidth: true
                    text: qsTrId("guide.noa.sandbox.header")
                    color: Theme.colors.text.heading
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.uiFontSize
                    font.weight: Font.DemiBold
                    elide: Text.ElideRight
                }

                Rectangle {
                    radius: Theme.smallControlRadius
                    color: Theme.colors.background.control
                    border.width: 1
                    border.color: root.stage === 1
                        ? Theme.colors.accent.primary
                        : Theme.colors.border.normal
                    implicitWidth: statusLabel.implicitWidth + 14
                    implicitHeight: 22

                    Text {
                        id: statusLabel
                        anchors.centerIn: parent
                        text: root.statusText
                        color: root.completed ? "#4CC38A" : Theme.colors.text.secondary
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.captionFontSize
                    }
                }
            }

            // Simulated Simai Editor surface
            Rectangle {
                id: editorBox
                Layout.fillWidth: true
                implicitHeight: codeColumn.implicitHeight + 20
                radius: Theme.controlRadius
                color: Theme.surfaceColor(Theme.colors.background.surface)
                border.width: 1
                border.color: Theme.colors.border.normal
                clip: true

                ColumnLayout {
                    id: codeColumn
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 10
                    spacing: 4

                    Repeater {
                        model: root.displayRows
                        delegate: RowLayout {
                            id: codeRow
                            required property var modelData
                            Layout.fillWidth: true
                            spacing: 8

                            // Candidate #2/#3 left accent indicator (matches NoaCopilotGhostOverlay)
                            Rectangle {
                                Layout.preferredWidth: 2
                                Layout.fillHeight: true
                                radius: 1
                                color: modelData.accentBar ? Theme.colors.accent.primary : "transparent"
                                opacity: 0.7
                            }

                            Text {
                                Layout.preferredWidth: 20
                                Layout.alignment: Qt.AlignTop
                                text: String(modelData.lineNumber)
                                horizontalAlignment: Text.AlignRight
                                color: Theme.colors.text.lineNumber
                                font.family: Theme.codeFont.family
                                font.pixelSize: Math.min(13, Math.max(11, Theme.codeFont.pixelSize))
                            }

                            Text {
                                Layout.fillWidth: true
                                textFormat: Text.RichText
                                wrapMode: Text.WrapAnywhere
                                font.family: Theme.codeFont.family
                                font.pixelSize: Math.min(13, Math.max(11, Theme.codeFont.pixelSize))
                                color: Theme.colors.text.editor
                                text: {
                                    const committedHtml = root.formatCommitted(modelData.committed)
                                    const caretColor = root.colorHex(Theme.colors.accent.primary)
                                    const caretHtml = modelData.caret
                                        ? ("<span style=\"color:" + (caretBlink.on ? caretColor : "transparent") + ";font-weight:bold;\">|</span>")
                                        : ""
                                    const ghostColor = root.colorHex(Theme.colors.text.disabled)
                                    const ghostHtml = modelData.ghost.length > 0
                                        ? ("<span style=\"color:" + ghostColor + ";\">" + root.escapeHtml(modelData.ghost) + "</span>")
                                        : ""
                                    if (!committedHtml && !caretHtml && !ghostHtml)
                                        return "&nbsp;"
                                    return committedHtml + caretHtml + ghostHtml
                                }
                            }
                        }
                    }
                }

                // Inline toast (e.g. pressing Down while Group 1 is still streaming)
                Rectangle {
                    visible: root.toastMessage.length > 0
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.margins: 8
                    radius: Theme.smallControlRadius
                    color: Theme.surfaceColor(Theme.colors.background.elevated)
                    border.width: 1
                    border.color: Theme.colors.syntax.warning
                    width: toastText.implicitWidth + 14
                    height: toastText.implicitHeight + 8

                    Text {
                        id: toastText
                        anchors.centerIn: parent
                        text: root.toastMessage
                        color: Theme.colors.text.primary
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.captionFontSize
                    }
                }
            }

            // Bottom interactive key progress bar
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6

                    Repeater {
                        model: root.stepSequence
                        delegate: RowLayout {
                            id: stepChip
                            required property string modelData
                            required property int index
                            readonly property bool done: index < root.completedSteps
                            readonly property bool current: !root.completed
                                && ((root.stage === 0 && index === 0)
                                    || (root.stage === 2 && index === 1)
                                    || (root.stage === 3 && index === 2)
                                    || (root.stage === 4 && index === 3)
                                    || (root.stage === 5 && index === 4)
                                    || (root.stage === 6 && index === 5)
                                    || (root.stage === 7 && index === 6))
                            spacing: 4

                            Rectangle {
                                radius: Theme.smallControlRadius
                                implicitWidth: Math.max(28, chipLabel.implicitWidth + 12)
                                implicitHeight: 22
                                color: stepChip.current
                                    ? Theme.colors.accent.primary
                                    : (stepChip.done
                                        ? Theme.contentOverlayColor(Theme.colors.accent.primary, 0.16)
                                        : Theme.colors.background.control)
                                border.width: 1
                                border.color: stepChip.current || stepChip.done
                                    ? Theme.colors.accent.primary
                                    : Theme.colors.border.normal

                                Text {
                                    id: chipLabel
                                    anchors.centerIn: parent
                                    text: stepChip.modelData
                                    color: stepChip.current
                                        ? Theme.colors.text.onAccent
                                        : (stepChip.done ? Theme.colors.accent.primary : Theme.colors.text.secondary)
                                    font.family: Theme.codeFont.family
                                    font.pixelSize: Theme.captionFontSize
                                    font.weight: Font.DemiBold
                                }
                            }

                            Text {
                                visible: stepChip.index < root.stepSequence.length - 1
                                text: "›"
                                color: Theme.colors.text.disabled
                                font.family: Theme.uiFont
                                font.pixelSize: Theme.secondaryFontSize
                            }
                        }
                    }

                    Item { Layout.fillWidth: true }

                    Text {
                        text: root.completedSteps + " / " + root.stepSequence.length
                        color: root.completed ? "#4CC38A" : Theme.colors.text.secondary
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.captionFontSize
                        font.weight: Font.DemiBold
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: root.instructionText
                    color: root.completed ? "#4CC38A" : Theme.colors.text.primary
                    font.family: Theme.uiFont
                    font.pixelSize: Theme.secondaryFontSize
                    wrapMode: Text.Wrap
                }
            }
        }
    }
}
