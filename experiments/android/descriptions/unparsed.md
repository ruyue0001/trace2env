# AgentWorldBench android, DroidBot-style sub-source ("unparsed")

An Android phone emulator (portrait, screen reported as 2340x1080) explored by a DroidBot-style GUI agent in
third-party apps from app stores (a point-of-sale app, fitness, water-reminder, consumer-protection, poster-design,
free-call, travel, games, ...). Every trajectory is one app; the task instruction is generic ("Predict the next Android
UI state after applying the action in app <package>"), the exploration logs in, fills forms with dummy text, taps
through menus and dialogs, scrolls, and sometimes restarts the app.

Observation (screen state) format: `**Current Phone State:**` with bullet lines `• **App:** <package>`,
`• **Activity:** <activity class>`, `• **State ID:** <hash>/<hash>` (a DroidBot structure hash of the screen: equal ids
mean the same screen structure), `• **Screen:** WxH`, then an indented `Accessibility Tree:` where each line is
`- Class [id="pkg:id/name"] [text="..."] [flags=clickable,focusable,...] bounds=x1,y1,x2,y2` and indentation gives
the view hierarchy. Only meaningful views are listed; ids are stable across screens of the same app while texts and
bounds change.

Actions are DroidBot command text: `touch <button|p|input id=N bound_box=x1,y1,x2,y2>TEXT</button|p|input>`,
`long_touch <...>`, `set_text <input ...>current text</input> dummy_user_input` (types the trailing text into the
field), `scroll up|down <scrollbar bound_box=...></scrollbar>`, `select` / `unselect <checkbox ...>` (toggle a
checkbox), `intent am start <package>/<activity>` (relaunches the app's start activity) and `kill_app` (stops the app;
the next state is typically the launcher or the app relaunched). `id=N` is DroidBot's view index and `bound_box` the
target view's bounds on the current screen.

Typical behaviour: touch on a button or row navigates (new Activity and State ID), opens a dialog, or does nothing for
disabled views; set_text keeps the screen and replaces the field's text with `dummy_user_input`; scroll keeps the
Activity and changes the visible subtree; intent / kill_app reset to the app's start screen (splash, login or main).
Each transition is teacher-forced: the next state is the real device state after the action.
