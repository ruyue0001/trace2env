# AgentWorldBench android, phone-tool sub-source ("phone")

An Android phone emulator (1080x2400 portrait) driven by an agent that calls a `phone` tool through `<invoke>` XML. The
traces are multi-step productivity tasks (file management in Files, messages in Mattermost and Mastodon, a mail client,
an OpenDocument reader, Chrome) whose observation is a "Current Phone State" summary.

Observation (screen state) format: `**Current Phone State:**` followed by bullet lines `• **App:** Name (package)`,
`• **Keyboard:** Hidden|Shown`, `• **Focused Element:** '...'`, then "Current Clickable UI elements from the device in
the schema 'index. className: resourceId, text - bounds(x1,y1,x2,y2)'" and one numbered line per element, e.g.
`8. TextView: "com.google.android.documentsui:id/breadcrumb_text", "Downloads" - (0,258,267,384)`. Elements without a
resource id show their class name or text in the quotes; icon-font glyphs appear as private-use characters. The index
is renumbered on every screen. React-Native and web-view apps (the mail client, Chrome pages) expose few resource ids.

Actions: `<invoke name="phone">` with `<parameter name="action">click|long_press|type|swipe|open_app|system_button
</parameter>` and parameters `index` (an element of the current screen), `text` (type, open_app), `coordinate` /
`coordinate2` / `duration` (swipe), `button` (system_button: back, home, enter). A `<invoke name="remember">` block may
precede the phone call to store a note for the agent; it changes nothing on the device.

Typical behaviour: click on a row or button opens the target screen or a dialog and may change the App line; type
fills the focused field (the keyboard line becomes Shown while typing, the focused element shows the field); open_app
switches the App line and the element list to that app's main screen; swipe scrolls the list, renumbering elements;
system_button back closes dialogs, keyboards or returns to the previous screen. Each transition is teacher-forced: the
next state is the real device state after the action.
