import json
import requests
from ex1 import *


def submit(email, token, name, outputs):
    query = json.dumps({
        'assignmentSlug': name,
        'submitterEmail': email,
        'secret': token,
        'parts': outputs
    })
    response = requests.post("https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1", data={"jsonBody": query})
    print(response.content)


def output(m):
    res = []
    if isinstance(m, np.ndarray):
        if isinstance(m[0], np.ndarray):
            for column in m.T:
                for item in column:
                    res.append("%.5f" % item)
        else:
            for item in m:
                res.append("%.5f" % item)
    else:
        return "%.5f" % m
    return ' '.join(res)


def generate_test_cases():
    np.set_printoptions(precision=5)
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    ones = np.ones(20)
    exps = np.exp(1) + np.exp(2) * np.arange(0.1, 2.1, 0.1)
    X1 = (np.column_stack((ones, exps)))
    Y1 = exps + np.sin(ones) + np.cos(exps)
    X2 = np.column_stack((X1, np.power(exps, 0.5), np.power(exps, 0.25)))
    Y2 = np.power(Y1, 0.5) + Y1

    return {
        "1": {"output": output(warmUpExercise())},
        "2": {"output": output(computeCost(X1, Y1, np.array([0.5 -0.5])))},
        "3": {"output": output(gradientDescent(X1, Y1, np.array([0.5 -0.5]), 0.01, 10))},
        "4": {"output": output(featureNormalize(X2))},
        "5": {"output": output(computeCostMulti(X2, Y2, np.array([0.1, 0.2, 0.3, 0.4])))},
        "6": {"output": output(gradientDescentMulti(X2, Y2, np.array([-0.1, -0.2, -0.3, -0.4]), 0.01, 10))},
        "7": {"output": output(normalEqn(X2, Y2))},
    }


test()
