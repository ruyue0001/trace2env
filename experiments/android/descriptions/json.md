# AgentWorldBench android, AndroidWorld-style sub-source ("json")

An Android phone emulator (Pixel-class device, 1080x2400 portrait, Android 13/14) driven by a UI agent through the
accessibility tree. The traces are AndroidWorld-style tasks on open-source apps installed on the device: Simple
Calendar Pro, Markor, Joplin, Broccoli (recipes), Simple SMS Messenger, Contacts, Settings, Files, Audio Recorder, Retro
Music, OpenTracks, Simple Draw Pro, Pro Expense, Clock, Camera, OsmAnd, Chrome, Gmail, Google Docs and a few news and
shopping apps; many episodes start on the launcher home screen (nexuslauncher workspace) and open an app first.

Observation (screen state) format, produced by the same formatter after every action: one line per visible UI element
in tree order, `[ i] | android.widget.Class | text="..." desc="..." res="pkg:id/name" | bounds=[x1,y1][x2,y2] | flags`
(flags among clickable, long_clickable, scrollable, focused, checked, selected, editable). Only visible elements appear;
the index `i` is the element's position in this list and is renumbered on every screen, so the same button can carry a
different index on two screens. System UI lines (status bar clock, wifi, battery, notifications under
com.android.systemui) close most screens; the keyboard adds com.google.android.inputmethod lines when a field is
focused.

Actions are JSON objects in the agent's action block. Two agents contributed traces. The AndroidWorld agent uses
`{"action_type": "click"|"long_press"|"input_text"|"scroll"|"open_app"|"navigate_back"|"navigate_home"|"keyboard_enter"|
"wait", ...}` with `index` naming an element of the current screen, `text` for input_text, `direction` (up/down/left/
right) for scroll, `app_name` for open_app. The second agent uses `{"action": "type"|"swipe"|"wait"|"click"|"long_press"
|"system_button", ...}` with screen coordinates (`coordinate`, `coordinate2`, `coordinate_abs`), `text` for type and
`time` for wait.

Typical behaviour: open_app replaces the launcher with the app's last or main screen (a first launch may show a
permission or onboarding dialog); click on a list row or button navigates or opens a dialog; input_text / type fills the
focused field and often keeps the keyboard open; scroll shifts the visible window of a list so indices and bounds
change while the screen identity (app, top bar, resource ids) stays; navigate_back returns to the previous screen or
closes a dialog or keyboard; long_press opens a context menu or selection mode; wait leaves the screen unchanged apart
from time-dependent text. Each transition is teacher-forced: the next screen is the real device state after the action.
