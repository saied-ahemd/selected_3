from flask import Flask, request, jsonify
from numpy import copy
import autocompleter
import autocompleter_copy

app = Flask(__name__)


@app.route('/autocomplete')
def autocomplete():
    """ Generate autocompletions given the query 'q' """

    q = request.args.get('q')
    completions = my_autocompleter.generate_completions(
        q, data_clean, model, tdidf_matrice)
    return jsonify({"Completions": completions})


if __name__ == "__main__":

    my_autocompleter = autocompleter_copy.Autocompleter()
    data_orig = autocompleter_copy.LoadingData()
    data = data_orig.load_data()
    data_clean = my_autocompleter.process_data(data)
    model, tdidf_matrice = my_autocompleter.calc_matrice(data_clean)
    print("ready to run...")

    app.run(host="0.0.0.0", port=80)
