pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Controls.impl as ControlsImpl
import QtQuick.Layouts
import MiaCode.UI

Popup {
    id: root

    required property var sections
    property int startIndex: 0
    property int selectedIndex: 0
    property int layoutRevision: 0
    property bool openingRange: false
    property string interactiveActiveKey: ""
    property string interactivePrompt: ""
    property int interactiveCompletedSteps: 0
    property int interactiveCurrentStepIndex: -1
    property var interactiveStepSequence: []
    readonly property bool confirmingSkip: skipPracticeDialog.visible
    property real dragOffsetX: 0
    property real dragOffsetY: 0
    readonly property var selectedSection: sections[selectedIndex]
    readonly property bool interactiveWorkspace: !!(selectedSection && selectedSection.interactiveWorkspace)
    readonly property rect spotlight: interactiveWorkspace ? Qt.rect(0, 0, 0, 0) : regionRect(selectedSection)
    readonly property bool usingFallback: !interactiveWorkspace && !itemAvailable(selectedSection.target)
    readonly property bool lastStep: selectedIndex === sections.length - 1
    readonly property int visibleStepNumber: selectedIndex - startIndex + 1
    readonly property int visibleStepCount: Math.max(1, sections.length - startIndex)
    readonly property int iconSize: Theme.compactControlHeight
    readonly property real cardDesiredWidth: Math.min(
        selectedSection && selectedSection.cardWidth ? selectedSection.cardWidth : 320,
        Math.max(240, (parent ? parent.width : 800) - 2 * Theme.dialogMargin))
    readonly property rect floatingCardRect: computeFloatingRect(cardDesiredWidth, card.desiredHeight)

    parent: Overlay.overlay
    popupType: Popup.Item
    x: interactiveWorkspace ? floatingCardRect.x : 0
    y: interactiveWorkspace ? floatingCardRect.y : 0
    width: interactiveWorkspace ? floatingCardRect.width : (parent ? parent.width : 0)
    height: interactiveWorkspace ? floatingCardRect.height : (parent ? parent.height : 0)
    padding: 0
    margins: 0
    modal: !interactiveWorkspace
    dim: false
    focus: !interactiveWorkspace
    closePolicy: interactiveWorkspace ? Popup.NoAutoClose : Popup.CloseOnEscape
    z: 100
    background: Item {}
    onAboutToShow: {
        if (!openingRange) {
            startIndex = 0
            selectedIndex = 0
        }
        openingRange = false
        dragOffsetX = 0
        dragOffsetY = 0
    }
    onOpened: {
        if (!interactiveWorkspace)
            nextButton.forceActiveFocus()
        bumpLayout()
    }
    onSelectedIndexChanged: {
        viewport.contentItem.contentY = 0
        dragOffsetX = 0
        dragOffsetY = 0
        if (visible && !interactiveWorkspace)
            nextButton.forceActiveFocus()
        bumpLayout()
    }

    function computeFloatingRect(cardWidth, desiredHeight) {
        const hostW = parent ? parent.width : 800
        const hostH = parent ? parent.height : 600
        const margin = Theme.dialogMargin
        const w = Math.min(cardWidth, Math.max(240, hostW - 2 * margin))
        const h = Math.min(Math.max(160, desiredHeight), Math.max(160, hostH - 2 * margin))
        let baseX = margin
        let baseY = hostH - margin - h
        const rev = root.layoutRevision
        const anchorItem = selectedSection ? selectedSection.target : null
        if (rev >= 0 && anchorItem && itemAvailable(anchorItem) && parent) {
            const origin = anchorItem.mapToItem(parent, 0, 0)
            baseX = origin.x + Theme.dialogPadding
            baseY = origin.y + anchorItem.height - h - Theme.dialogPadding
        }
        return Qt.rect(
            bounded(baseX + dragOffsetX, margin, Math.max(margin, hostW - margin - w)),
            bounded(baseY + dragOffsetY, margin, Math.max(margin, hostH - margin - h)),
            w,
            h
        )
    }

    function openFromId(sectionId) {
        let index = -1
        for (let i = 0; i < sections.length; ++i) {
            if (sections[i].id === sectionId) {
                index = i
                break
            }
        }
        if (index < 0)
            return
        openingRange = true
        startIndex = index
        selectedIndex = index
        open()
    }

    function bumpLayout() {
        layoutRevision += 1
    }

    function dispatchSectionKey(event) {
        if (!selectedSection || !selectedSection.keyHandler)
            return false
        if (event.modifiers !== Qt.NoModifier)
            return false
        if (event.key === Qt.Key_Tab || event.key === Qt.Key_Down
                || event.key === Qt.Key_Up || event.key === Qt.Key_Right) {
            if (selectedSection.keyHandler(event.key)) {
                event.accepted = true
                return true
            }
        }
        return false
    }

    function itemAvailable(item) {
        const rev = root.layoutRevision
        if (rev < 0 || !item || !root.parent || item.Window.window !== root.parent.Window.window)
            return false
        if (!item.visible || item.width <= 0 || item.height <= 0)
            return false
        for (let node = item.parent; node; node = node.parent) {
            if (!node.visible)
                return false
            // Qt Quick Flickable.contentItem reports 0x0 even while its viewport and children are visible.
            const isFlickableContent = node.parent && node.parent.contentItem === node
            if (!isFlickableContent && (node.width <= 0 || node.height <= 0))
                return false
        }
        return true
    }

    function regionRect(section) {
        if (!root.visible || !section)
            return Qt.rect(0, 0, 0, 0)
        if (!itemAvailable(section.target))
            return itemRect(section.fallbackTarget, 0)
        const primary = itemRect(section.target, section.focusHeight || 0)
        if (!itemAvailable(section.secondaryTarget))
            return primary
        const secondary = itemRect(section.secondaryTarget, 0)
        const left = Math.min(primary.x, secondary.x)
        const top = Math.min(primary.y, secondary.y)
        const right = Math.max(primary.x + primary.width, secondary.x + secondary.width)
        const bottom = Math.max(primary.y + primary.height, secondary.y + secondary.height)
        return Qt.rect(left, top, right - left, bottom - top)
    }

    function itemRect(item, heightLimit) {
        const rev = root.layoutRevision
        if (rev < 0 || !itemAvailable(item))
            return Qt.rect(0, 0, 0, 0)
        // 读取祖先几何，使高亮跟随分栏、滚动和窗口缩放。
        for (let node = item; node; node = node.parent) {
            node.x
            node.y
            node.width
            node.height
        }
        const origin = item.mapToItem(surface, 0, 0)
        const focusHeight = heightLimit > 0 ? Math.min(item.height, heightLimit) : item.height
        const inset = Theme.chromePadding
        let left = Math.max(0, origin.x - inset)
        let top = Math.max(0, origin.y - inset)
        let right = Math.min(surface.width, origin.x + item.width + inset)
        let bottom = Math.min(surface.height, origin.y + focusHeight + inset)
        for (let node = item.parent; node; node = node.parent) {
            if (node.clip) {
                const clipOrigin = node.mapToItem(surface, 0, 0)
                left = Math.max(left, clipOrigin.x)
                top = Math.max(top, clipOrigin.y)
                right = Math.min(right, clipOrigin.x + node.width)
                bottom = Math.min(bottom, clipOrigin.y + node.height)
            }
        }
        return Qt.rect(left, top, Math.max(0, right - left), Math.max(0, bottom - top))
    }

    function bounded(value, minimum, maximum) {
        return Math.max(minimum, Math.min(value, maximum))
    }

    function placeCard(cardWidth, desiredHeight) {
        const margin = Theme.dialogMargin
        const gap = Theme.dialogPadding
        const region = root.spotlight
        const availableHeight = surface.height - 2 * margin
        const cardHeight = Math.min(desiredHeight, availableHeight)
        const preferred = root.usingFallback ? "below" : root.selectedSection.placement
        const alignEnd = root.usingFallback || root.selectedSection.alignment === "end"
        const spaces = {
            below: surface.height - margin - region.y - region.height - gap,
            above: region.y - gap - margin,
            right: surface.width - margin - region.x - region.width - gap,
            left: region.x - gap - margin
        }
        const sides = [preferred, "below", "right", "above", "left"]
        let side = ""
        for (const candidate of sides) {
            const requiredSpace = candidate === "below" || candidate === "above"
                ? cardHeight : cardWidth
            if (spaces[candidate] >= requiredSpace) {
                side = candidate
                break
            }
        }
        // 短窗口使用上下空隙，说明正文滚动，按钮保留在卡片底部。
        if (side === "")
            side = spaces.below >= spaces.above ? "below" : "above"
        const height = side === "below" || side === "above"
            ? Math.min(cardHeight, Math.max(0, spaces[side])) : cardHeight
        let x = alignEnd ? region.x + region.width - cardWidth : region.x
        let y = region.y
        if (side === "below")
            y = region.y + region.height + gap
        else if (side === "above")
            y = region.y - gap - height
        else if (side === "right")
            x = region.x + region.width + gap
        else
            x = region.x - gap - cardWidth
        return {
            x: bounded(x, margin, surface.width - margin - cardWidth),
            y: bounded(y, margin, surface.height - margin - height),
            height: height,
            side: side
        }
    }

    function advance() {
        if (interactiveWorkspace && interactiveCompletedSteps < interactiveStepSequence.length)
            skipPracticeDialog.open()
        else if (lastStep)
            close()
        else
            selectedIndex += 1
    }

    ChoiceDialog {
        id: skipPracticeDialog
        objectName: "skipPracticeDialog"
        z: root.z + 1
        title: "NoaCopilot"
        message: qsTrId("guide.noa.continuation.confirm_skip")
        choices: [
            { id: "cancel", label: qsTrId("action.cancel"), role: "reject" },
            { id: "skip", label: qsTrId("action.ok"), role: "accept" }
        ]
        dismissChoiceId: "cancel"
        onChosen: choiceId => {
            if (choiceId === "skip" && root.visible && root.interactiveWorkspace)
                root.selectedIndex += 1
        }
    }

    contentItem: Item {
        id: surface
        Accessible.name: qsTrId("guide.page_one")
        Keys.onTabPressed: event => root.dispatchSectionKey(event)
        Keys.onPressed: event => root.dispatchSectionKey(event)

        Rectangle {
            visible: !root.interactiveWorkspace
            width: parent.width
            height: root.spotlight.y
            color: Theme.modalScrimColor
        }
        Rectangle {
            visible: !root.interactiveWorkspace
            y: root.spotlight.y
            width: root.spotlight.x
            height: root.spotlight.height
            color: Theme.modalScrimColor
        }
        Rectangle {
            visible: !root.interactiveWorkspace
            x: root.spotlight.x + root.spotlight.width
            y: root.spotlight.y
            width: parent.width - x
            height: root.spotlight.height
            color: Theme.modalScrimColor
        }
        Rectangle {
            visible: !root.interactiveWorkspace
            y: root.spotlight.y + root.spotlight.height
            width: parent.width
            height: parent.height - y
            color: Theme.modalScrimColor
        }

        MouseArea {
            anchors.fill: parent
            visible: !root.interactiveWorkspace
            acceptedButtons: Qt.AllButtons
            onWheel: wheel => wheel.accepted = true
        }

        Rectangle {
            x: root.spotlight.x
            y: root.spotlight.y
            width: root.spotlight.width
            height: root.spotlight.height
            visible: !root.interactiveWorkspace && width > 0 && height > 0
            color: "transparent"
            radius: Theme.controlRadius
            border.width: 2
            border.color: Theme.colors.accent.primary
        }

        Item {
            id: card
            readonly property real padding: Theme.dialogPadding
            readonly property real desiredHeight: header.implicitHeight
                + bodyColumn.implicitHeight + footer.implicitHeight
                + 2 * padding + 2 * Theme.dialogPadding
            readonly property var placement: root.interactiveWorkspace
                ? ({ x: 0, y: 0, height: surface.height, side: "below" })
                : root.placeCard(root.cardDesiredWidth, desiredHeight)
            width: root.interactiveWorkspace ? surface.width : root.cardDesiredWidth
            height: root.interactiveWorkspace ? surface.height : placement.height
            x: root.interactiveWorkspace ? 0 : placement.x
            y: root.interactiveWorkspace ? 0 : placement.y

            // 指示角位于卡片边缘，序号放在卡片内，控件图标保持可见。
            Rectangle {
                readonly property bool verticalSide: card.placement.side === "below"
                    || card.placement.side === "above"
                width: Theme.dialogPadding
                height: width
                rotation: 45
                x: verticalSide
                    ? root.bounded(root.spotlight.x + root.spotlight.width / 2 - card.x,
                        2 * card.padding, card.width - 2 * card.padding) - width / 2
                    : card.placement.side === "right" ? -width / 2 : card.width - width / 2
                y: verticalSide
                    ? card.placement.side === "below" ? -height / 2 : card.height - height / 2
                    : root.bounded(root.spotlight.y + Theme.controlMinHeight / 2 - card.y,
                        2 * card.padding, card.height - 2 * card.padding) - height / 2
                color: Theme.colors.background.elevated
                border.color: Theme.floatingBorderColor
                border.width: 1
                visible: !root.interactiveWorkspace && root.spotlight.width > 0 && root.spotlight.height > 0
            }

            FloatingCard {
                anchors.fill: parent
                cornerRadius: Theme.popupRadius
                tintColor: Theme.colors.background.elevated
                shadowOpacity: Theme.dialogShadowOpacity
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: card.padding
                spacing: Theme.dialogPadding

                ColumnLayout {
                    id: header
                    Layout.fillWidth: true
                    spacing: Theme.panelPadding

                    RowLayout {
                        Layout.fillWidth: true
                        Rectangle {
                            Layout.preferredWidth: Theme.controlMinHeight
                            Layout.preferredHeight: Theme.controlMinHeight
                            radius: width / 2
                            color: Theme.colors.accent.primary
                            Text {
                                anchors.centerIn: parent
                                text: root.visibleStepNumber
                                color: Theme.colors.text.onAccent
                                font.family: Theme.uiFont
                                font.pixelSize: Theme.uiFontSize
                                font.weight: Font.DemiBold
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: root.selectedSection.title
                            color: Theme.colors.text.heading
                            font.family: Theme.uiFont
                            font.pixelSize: Theme.headingFontSize
                            font.weight: Font.DemiBold
                            wrapMode: Text.Wrap
                            MouseArea {
                                anchors.fill: parent
                                visible: root.interactiveWorkspace
                                cursorShape: Qt.SizeAllCursor
                                property point pressPos: Qt.point(0, 0)
                                onPressed: mouse => {
                                    pressPos = Qt.point(mouse.x, mouse.y)
                                }
                                onPositionChanged: mouse => {
                                    if (!pressed)
                                        return
                                    root.dragOffsetX += mouse.x - pressPos.x
                                    root.dragOffsetY += mouse.y - pressPos.y
                                }
                            }
                        }
                        IconButton {
                            iconSource: Qt.resolvedUrl("icons/close.svg")
                            tooltip: qsTrId("guide.exit")
                            Accessible.name: tooltip
                            Keys.onTabPressed: event => root.dispatchSectionKey(event)
                            Keys.onPressed: event => root.dispatchSectionKey(event)
                            onClicked: root.close()
                        }
                    }

                }

                ScrollView {
                    id: viewport
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    contentWidth: availableWidth
                    contentHeight: bodyColumn.implicitHeight
                    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
                    ScrollBar.vertical: AppScrollBar {}

                    Column {
                        id: bodyColumn
                        width: viewport.availableWidth
                        spacing: Theme.panelPadding
                        Column {
                            width: parent.width
                            spacing: 6
                            visible: root.interactiveWorkspace && root.interactiveStepSequence.length > 0

                            Row {
                                width: parent.width
                                spacing: 4
                                Repeater {
                                    model: root.interactiveStepSequence
                                    delegate: Row {
                                        id: stepPill
                                        required property string modelData
                                        required property int index
                                        readonly property bool done: index < root.interactiveCompletedSteps
                                        readonly property bool current: index === root.interactiveCurrentStepIndex
                                        spacing: 4

                                        Rectangle {
                                            radius: Theme.smallControlRadius
                                            width: Math.max(26, pillText.implicitWidth + 10)
                                            height: 20
                                            color: stepPill.current
                                                ? Theme.colors.accent.primary
                                                : (stepPill.done
                                                    ? Theme.contentOverlayColor(Theme.colors.accent.primary, 0.16)
                                                    : Theme.colors.background.control)
                                            border.width: 1
                                            border.color: stepPill.current || stepPill.done
                                                ? Theme.colors.accent.primary
                                                : Theme.colors.border.normal

                                            Text {
                                                id: pillText
                                                anchors.centerIn: parent
                                                text: stepPill.modelData
                                                color: stepPill.current
                                                    ? Theme.colors.text.onAccent
                                                    : (stepPill.done ? Theme.colors.accent.primary : Theme.colors.text.secondary)
                                                font.family: Theme.codeFont.family
                                                font.pixelSize: Theme.captionFontSize
                                                font.weight: Font.DemiBold
                                            }
                                        }

                                        Text {
                                            visible: stepPill.index < root.interactiveStepSequence.length - 1
                                            anchors.verticalCenter: parent.verticalCenter
                                            text: "›"
                                            color: Theme.colors.text.disabled
                                            font.family: Theme.uiFont
                                            font.pixelSize: Theme.captionFontSize
                                        }
                                    }
                                }
                            }

                            Text {
                                width: parent.width
                                text: root.interactivePrompt
                                color: root.interactiveCompletedSteps >= root.interactiveStepSequence.length
                                    ? Theme.colors.status.success
                                    : Theme.colors.text.primary
                                font.family: Theme.uiFont
                                font.pixelSize: Theme.secondaryFontSize
                                wrapMode: Text.Wrap
                            }
                        }
                        Flow {
                            width: parent.width
                            spacing: Theme.panelPadding
                            visible: (root.selectedSection.icons || []).length > 0

                            Repeater {
                                model: root.selectedSection.icons || []
                                delegate: ControlsImpl.IconImage {
                                    required property url modelData
                                    width: root.iconSize
                                    height: root.iconSize
                                    source: modelData
                                    sourceSize: Theme.iconSourceSize(width, height)
                                    color: Theme.colors.text.active
                                }
                            }
                        }
                        GridLayout {
                            width: parent.width
                            columns: root.selectedSection.columns || 2
                            uniformCellWidths: true
                            uniformCellHeights: true
                            columnSpacing: Theme.panelPadding
                            rowSpacing: Theme.panelPadding

                            Repeater {
                                model: root.selectedSection.features
                                delegate: Item {
                                    id: feature
                                    required property var modelData
                                    readonly property var icons: modelData.icons
                                        || (modelData.icon ? [modelData.icon] : [])
                                    readonly property bool isKey: modelData.kind === "key"
                                    readonly property bool activeKey: isKey
                                        && root.interactiveActiveKey.length > 0
                                        && root.interactiveActiveKey === (modelData.sample || "")
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.columnSpan: modelData.span || 1
                                    implicitHeight: Math.max(Theme.activityButtonSize,
                                        featureRow.implicitHeight + 2 * Theme.panelPadding)
                                    Accessible.role: Accessible.StaticText
                                    Accessible.name: modelData.label

                                    Rectangle {
                                        anchors.fill: parent
                                        radius: Theme.controlRadius
                                        color: feature.activeKey
                                            ? Theme.contentOverlayColor(Theme.colors.accent.primary, 0.14)
                                            : Theme.colors.background.control
                                        border.color: feature.activeKey
                                            ? Theme.colors.accent.primary
                                            : Theme.colors.border.normal
                                    }
                                    RowLayout {
                                        id: featureRow
                                        anchors.fill: parent
                                        anchors.margins: Theme.panelPadding
                                        spacing: Theme.panelPadding

                                        Item {
                                            visible: feature.icons.length > 0 || sample.visible
                                            Layout.preferredWidth: Math.max(root.iconSize,
                                                featureIcons.implicitWidth,
                                                sample.visible ? sample.implicitWidth + (feature.isKey ? 12 : 0) : 0)
                                            Layout.preferredHeight: root.iconSize
                                            Rectangle {
                                                anchors.fill: parent
                                                visible: feature.isKey
                                                radius: Theme.smallControlRadius
                                                color: feature.activeKey
                                                    ? Theme.colors.accent.primary
                                                    : Theme.colors.background.elevated
                                                border.width: 1
                                                border.color: feature.activeKey
                                                    ? Theme.colors.accent.primary
                                                    : Theme.colors.border.normal
                                            }
                                            Row {
                                                id: featureIcons
                                                anchors.centerIn: parent
                                                spacing: Theme.chromePadding
                                                Repeater {
                                                    model: feature.icons
                                                    delegate: ControlsImpl.IconImage {
                                                        required property url modelData
                                                        width: root.iconSize
                                                        height: width
                                                        sourceSize: Theme.iconSourceSize(width, height)
                                                        source: modelData
                                                        color: Theme.colors.text.active
                                                    }
                                                }
                                            }
                                            Text {
                                                id: sample
                                                anchors.centerIn: parent
                                                visible: feature.modelData.sample !== undefined
                                                text: feature.modelData.sample || ""
                                                color: feature.activeKey
                                                    ? Theme.colors.text.onAccent
                                                    : (feature.modelData.kind === "code"
                                                        ? Theme.colors.syntax.duration : Theme.colors.text.active)
                                                font.family: (feature.modelData.kind === "code" || feature.isKey)
                                                    ? Theme.codeFont.family : Theme.uiFont
                                                font.pixelSize: Theme.uiFontSize
                                                font.weight: Font.DemiBold
                                            }
                                        }
                                        Text {
                                            Layout.fillWidth: true
                                            text: feature.modelData.label
                                            color: Theme.colors.text.primary
                                            font.family: Theme.uiFont
                                            font.pixelSize: Theme.secondaryFontSize
                                            wrapMode: Text.Wrap
                                        }
                                    }
                                    MouseArea {
                                        anchors.fill: parent
                                        visible: feature.isKey && !!root.selectedSection.symbolHandler
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: root.selectedSection.symbolHandler(feature.modelData.sample || "")
                                    }
                                }
                            }
                        }
                        Text {
                            width: parent.width
                            visible: root.usingFallback
                            text: root.usingFallback ? (root.selectedSection.hint || "") : ""
                            color: Theme.colors.text.secondary
                            font.family: Theme.uiFont
                            font.pixelSize: Theme.secondaryFontSize
                            wrapMode: Text.Wrap
                        }
                    }
                }

                RowLayout {
                    id: footer
                    Layout.fillWidth: true
                    spacing: Theme.panelPadding
                    AppButton {
                        text: qsTrId("guide.previous")
                        enabled: root.selectedIndex > root.startIndex
                        Keys.onTabPressed: event => root.dispatchSectionKey(event)
                        Keys.onPressed: event => root.dispatchSectionKey(event)
                        onClicked: root.selectedIndex -= 1
                    }
                    Text {
                        Layout.fillWidth: true
                        text: qsTrId("guide.step_counter")
                            .arg(root.visibleStepNumber).arg(root.visibleStepCount)
                        color: Theme.colors.text.secondary
                        font.family: Theme.uiFont
                        font.pixelSize: Theme.secondaryFontSize
                        horizontalAlignment: Text.AlignHCenter
                    }
                    AppButton {
                        id: nextButton
                        text: root.lastStep ? qsTrId("guide.finish") : qsTrId("guide.next")
                        emphasized: true
                        Keys.onTabPressed: event => root.dispatchSectionKey(event)
                        Keys.onPressed: event => root.dispatchSectionKey(event)
                        onClicked: root.advance()
                    }
                }
            }
        }
    }
}
