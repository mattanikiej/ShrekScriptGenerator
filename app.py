from flask import Flask, render_template, request, redirect, url_for
from script_generator import ScriptGenerator

# initialize web app
app = Flask(__name__)

# initialize generator
generator = ScriptGenerator()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        script_title = request.form['script title']
        first_words = request.form['first words']

        generator.set_script_title(script_title)
        generator.set_first_words(first_words)

        return redirect(url_for('script'))

    return render_template("home.html")


@app.route('/script', methods=['GET', 'POST'])
def script():
    amazing_custom_script = generator.generate_script()
    return render_template("script.html", movie_script=amazing_custom_script)


if __name__ == '__main__':
    app.run()
