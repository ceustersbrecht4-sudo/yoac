# Adding the intro page and cookie banner to Player Tracker

Three steps, all inside `C:\Users\47289\Downloads\player_tracker_web`.

## 1. Copy the two new files into the `templates` folder

Put them next to your existing `index.html`:

- `templates/landing.html`: the intro page with the **Get Started** button
- `templates/_cookie_banner.html`: the cookie consent banner

## 2. Change 3 lines in `app.py`

Find this (around line 140):

```python
@app.route("/")
def index():
    return render_template("index.html")
```

Replace it with this:

```python
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/upload")
def index():
    return render_template("index.html")
```

That's the only change. Uploads still go to `/analyze` as before, so the upload tool keeps
working. It's just at `/upload` now.

## 3. (Optional) Show the cookie banner on the upload page too

Open `templates/index.html` in Notepad. Just before the `</body>` line near the bottom, add:

```html
{% include "_cookie_banner.html" %}
```

After someone clicks Accept or Decline, the banner stays hidden on every page for a year.

## Run it

```cmd
cd C:\Users\47289\Downloads\player_tracker_web
py app.py
```

Open http://127.0.0.1:5000. You'll see the intro page first, and **Get Started** takes you
to the upload tool.
