Environment: four self-hosted WebArena websites operated through a real Chromium browser via the
Playwright MCP tools. Each episode is one task-agent run on one WebArena task: the browser starts
logged out on a blank page, the agent navigates, logs in with the site's account when the task needs
it, reads pages and fills forms, and stops with an answer or once the requested change is made.

The sites, always under these hostnames (no ports):
- http://gitlab.example.com — GitLab CE (projects, issues, merge requests, members, milestones,
  groups, user profiles; sign-in at /users/sign_in, account byteblaze / hello1234).
- http://magento-store.example.com — "One Stop Market", a Magento storefront (categories, product
  pages, search, cart, wishlist, customer account with order history, contact-us form; sign-in at
  /customer/account/login/, account emma.lopez@gmail.com / Password.1).
- http://magento-admin.example.com/admin — the Magento admin panel of that store (dashboard, sales
  orders and invoices, catalog products, customers, reviews, marketing cart/catalog price rules,
  content pages, reports; account admin / admin1234).
- http://forum.example.com — a Postmill forum (Reddit-like: forums under /f/<name>, posts,
  comments, up/down votes, user pages, submit/create forms; /login, account MarvelsGrantMan136 /
  test1234).
All four are the same populated database snapshot in every episode: product names and prices,
order numbers, customers, projects, issues, users, forums and posts are facts about the environment
that recur across episodes, not task-specific fixtures. Tasks ran one after another on shared
instances, so an episode may show an object (a forum, a project, a milestone, a cancelled order)
that an earlier episode created or changed.

Actions. One action = one Playwright MCP tool call; the action type is the tool name and the
arguments are the call's JSON arguments: browser_navigate(url), browser_click(element, ref),
browser_type(element, ref, text[, submit][, slowly]), browser_fill_form(fields=[{name, type, ref,
value}, ...]), browser_press_key(key), browser_select_option(element, ref, values),
browser_evaluate(function[, element, ref]), browser_navigate_back(), browser_wait_for(time | text |
textGone). `ref` values (e12, e97, ...) are element handles valid only for the snapshot they came
from, and `element` is the agent's own description of the target; neither is state. What the
action does is determined by which page element the ref pointed to (a link with a `/url`, a
button, a textbox, a combobox) — the snapshot in the preceding observation says which.

Observations. The tool's text result, verbatim, made of `###` sections:
- `### Ran Playwright code` with a ```js fence holding the Playwright call the server executed
  (`await page.goto('...')`, `await page.getByRole('link', { name: '...' }).click()`,
  `await page.getByRole('textbox', { name: '...' }).fill('...')`, `await page.keyboard.press('...')`,
  `await page.evaluate('...')`, `await page.goBack()`).
- `### Page` with `- Page URL: ...`, `- Page Title: ...` and, when console messages exist,
  `- Console: N errors, M warnings`.
- `### Snapshot` with a ```yaml accessibility tree of the page: nested `- role "name" [ref=eN]
  [attributes]:` lines (link, button, textbox, combobox, heading [level=N], listitem, cell, img,
  text: ..., generic), `- /url: ...` under links, `[cursor=pointer]`, `[active]`, `[checked]`,
  `[expanded]`, `[disabled]`. This body is the page's content and structure: which elements exist,
  their labels, links and current values.
- `### Events` with `- New console entries: <path>/console-<timestamp>.log#L<range>` and inline
  `- [ERROR] ...` / `- [LOG] ...` lines; `### Result` with the JSON value of a browser_evaluate
  call; `### Error` with a Playwright error and its call log (timeouts, unsafe ports, failed
  navigation); `### Modal state` listing an open dialog that blocks other tools.
browser_navigate, browser_click and browser_navigate_back results carry the code section, the page
header and the snapshot of the page after the action; browser_type, browser_fill_form,
browser_press_key and browser_select_option results carry only the code section unless the agent
looked at the page afterwards, in which case the recorded observation is that look: a snapshot-style
observation that starts with `### Page` (no code section) and shows the page after the action, with
typed values inside the textboxes. browser_wait_for returns the code section and a snapshot.

State that matters:
- surface.page (object): {"url", "title", "site"} of the current page after the action; the snapshot
  body itself is rendered, not stored; surface.page.errors (list) for `### Error` results.
- session.site (string: gitlab | store | admin | forum), session.logged_in (object: site -> account
  when the sign-in succeeded), session.form (object: field name -> value typed but not yet
  submitted), session.history (list of visited URLs when it explains browser_navigate_back).
- world.gitlab (object): "namespace/project" -> {"visibility", "description", "issues": id ->
  {title, labels, assignee, state}, "members": username -> role, "milestones", "branches",
  "clone_ssh"}; world.gitlab_users (object): username -> {name, email, location}; world.gitlab_groups.
- world.store (object): "products": url or name -> {price, rating, category, options},
  "categories", "cart": [...], "wishlist": [...], "orders": number -> {date, status, total, items},
  "contact_messages".
- world.admin (object): "orders": number -> {customer, status, grand_total, date, items},
  "invoices", "products": sku or name -> {price, quantity, options, description}, "customers",
  "reviews": id -> {product, status, rating, text}, "cart_price_rules", "catalog_price_rules",
  "cms_pages": identifier -> {title, status}, "reports".
- world.forum (object): "forums": name -> {description, posts: id -> {title, author, score,
  comments}}, "users": name -> {posts, comments}, "votes".
Content a page reveals (a listing, an order total, an issue's labels, a product's price) is recorded
under the matching world.* map exactly as shown; refs, element descriptions and values the agent
computed are not state. Writes happen through forms: a click on a submit button (or a
browser_type with submit) after the form fields were filled changes world.* and usually navigates
to a confirmation page or back to a listing that shows the new state.

Scope. Rules describe what each site does in response to an action on a kind of page: where a link
leads, what a search or filter shows, what a form submission creates or changes and how the page
confirms it, what a failed sign-in or an invalid form shows, which pages require being signed in
(they redirect to the sign-in page). Renderers describe the section structure of the observation
kinds above; the snapshot body is page-specific content that comes from state and the page kind,
not from a fixed template.
