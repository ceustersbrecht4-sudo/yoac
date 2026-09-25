# Adding the intro page and cookie banner to Player Tracker

These files are made to be dropped into your existing
`C:\Users\47289\Downloads\player_tracker_web` folder.

## 1. Copy the templates

Copy both files into your project's `templates` folder (create it if it doesn't exist):

- `templates/landing.html`: the intro page with the **Get Started** button
- `templates/_cookie_banner.html`: the cookie consent banner

## 2. Change `app.py`

Your upload page is currently at `/`. Move it to `/upload` and put the intro page at `/`.

Find the route for the upload page. It looks something like this:

```python
@app.route("/", methods=["GET", "POST"])
def index():
    ...
```

Change it to this. Only the decorator line changes, and the function body stays the same:

```python
@app.route("/upload", methods=["GET", "POST"], endpoint="upload_page")
def index():
    ...
```

Keep whatever `methods=[...]` your route already has. `endpoint="upload_page"` is the
name the **Get Started** button links to.

Then add the new intro route. Put it right above the upload route:

```python
@app.route("/")
def landing():
    return render_template("landing.html")
```

Make sure `render_template` is in your Flask import at the top of `app.py`:

```python
from flask import Flask, render_template  # plus whatever else is already imported
```

## 3. Check the upload form

Open the HTML template for your upload page and find the `<form>` tag. If it says
`action="/"`, change it to `action="/upload"`. Otherwise uploads would go to the intro page
and fail. If there is no `action` at all, or it already uses `url_for(...)`, you don't need
to change anything.

## 4. Show the cookie banner on the upload page too (optional)

The banner appears on the intro page automatically. To show it on the upload page as well,
add this line just before `</body>` in that page's template:

```html
{% include "_cookie_banner.html" %}
```

After someone clicks Accept or Decline, the banner is hidden on every page for a year.
The choice is stored in a `cookie_consent` cookie.

## 5. Run it

```cmd
cd C:\Users\47289\Downloads\player_tracker_web
python app.py
```

Open http://127.0.0.1:5000. You should see the intro page, and **Get Started** takes you to
the upload tool.
