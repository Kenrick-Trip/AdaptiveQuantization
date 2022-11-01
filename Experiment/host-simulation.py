from flask import Flask

app = Flask(__name__)

@app.route('/newjob', methods = ['GET'])
def run_simulation():
    # Perform inference
    return "Finished job!"

if __name__ == '__main__':
    app.run()