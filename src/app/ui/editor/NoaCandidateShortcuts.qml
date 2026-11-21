import QtQuick

Item {
    id: root
    required property var noaCopilot

    Shortcut {
        sequence: "Up"
        context: Qt.WindowShortcut
        enabled: root.enabled && !!root.noaCopilot?.candidateSessionActive
        onActivated: root.noaCopilot.handleCandidateKey(Qt.Key_Up)
    }
    Shortcut {
        sequence: "Down"
        context: Qt.WindowShortcut
        enabled: root.enabled && !!root.noaCopilot?.candidateSessionActive
        onActivated: root.noaCopilot.handleCandidateKey(Qt.Key_Down)
    }
    Shortcut {
        sequence: "Tab"
        context: Qt.WindowShortcut
        enabled: root.enabled && !!root.noaCopilot?.candidateSessionActive
        autoRepeat: false
        onActivated: root.noaCopilot.handleCandidateKey(Qt.Key_Tab)
    }
}
