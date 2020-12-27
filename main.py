import os
import numpy
import argparse
import matplotlib.pyplot as pyplot

from scipy.stats import chisquare, norm, uniform, gamma, expon


def get_file_path():
    parser = argparse.ArgumentParser(description="Лабораторна робота 1, Іллін Дмитро, ІП-04мп")
    
    parser.add_argument("--path", type=str, help="Provide a relative path to the data file")

    args = parser.parse_args()

    return args.path


def read_file(file_path):
    if os.path.isdir(file_path):
        raise Exception("Directory provided. Path to file is required")

    if not os.path.exists(file_path):
        raise Exception("Invalid path provided")

    return numpy.genfromtxt(file_path, None,"#", ",")

def get_average_time_to_failure(data):
    return numpy.mean(data)

def get_standard_deviation(data):
    return numpy.std(data)

def get_stats_probability_of_failure(data):
    calculated = numpy.array([])
    count = len(data)

    for index in range(len(data)):
        calculated = numpy.append(calculated, index / count)
    
    return calculated

def get_histogram(data):
    return numpy.histogram(data, bins="auto", density=True)


def get_linspace(data):
    return numpy.linspace(data[0], data[-1], data.max())


def get_chi_square_test(distribution_to_test, ddofs, x, observed_value):
    expected = []

    for distribution in distribution_to_test:
        expected.append(distribution.pdf(x))

    chi_test = []

    for expected_value, ddof in zip(expected, ddofs):
        chi_test.append(chisquare(observed_value, expected_value, ddof=ddof))

    chi_value = [] 

    for statistic, pvalue in chi_test:
        if pvalue > 0.05:
            chi_value.append(statistic)
    
    found_distribution = distribution_to_test[chi_value.index(min(chi_value))]

    return found_distribution


def draw_f_chart(data, space, ft):
    pyplot.figure()
    pyplot.title('f(t) - f*(t)')
    pyplot.hist(data, bins='auto', density=True, label='f*(t)')
    pyplot.plot(space, ft, label='f(t)')
    pyplot.legend()
    pyplot.show(block=False)

def draw_q_chart(data, space, qxt, qt):
    pyplot.figure()
    pyplot.title('Q(t) - Q*(t)')
    pyplot.step(data, qxt,  label='Q*(t)')
    pyplot.fill_between(data, qxt, 0, step='pre')
    pyplot.plot(space, qt, label='Q(t)')
    pyplot.legend()
    pyplot.show(block=False)

def draw_p_chart(data, space, pxt, pt):
    pyplot.figure()
    pyplot.title('P(t) - P*(t)')
    pyplot.step(data, pxt, label='P*(t)')
    pyplot.fill_between(data, pxt)
    pyplot.plot(space, pt, label='P(t)')
    pyplot.legend()
    pyplot.show(block=False)

def draw_lambda_chart(space, lt):
    pyplot.figure()
    pyplot.title("Lambda(t)")
    pyplot.plot(space, lt, label="lambda(t)")
    pyplot.legend()
    pyplot.show(block=True)



def app():
    calculations = {}

    file_path = get_file_path()
    data = read_file(file_path)
    sorted_data = numpy.unique(data)

    calculations["Avg"] = get_average_time_to_failure(data)
    calculations["Std"] = get_standard_deviation(data)
    calculations["Q*(t)"] = get_stats_probability_of_failure(sorted_data)
    calculations["P*(t)"] = 1 - calculations["Q*(t)"]
    histogram, bin_edges = get_histogram(data)

    distribution_to_test = [
        expon(0, calculations["Avg"]), 
        uniform(0, data.max()),
        norm(calculations["Avg"], calculations["Std"]),
        gamma(*gamma.fit(data))
    ]

    found_distribution = get_chi_square_test(
        distribution_to_test, 
        [1,1,2,3],
        (bin_edges[1:] + bin_edges[:-1]) / 2, 
        histogram
    )

    linspace = get_linspace(sorted_data)

    calculations["f(t)"] = found_distribution.pdf(linspace)
    calculations["Q(t)"] = found_distribution.cdf(linspace)
    calculations["P(t)"] = found_distribution.sf(linspace)
    calculations["Lambda(t)"] = calculations["f(t)"] / calculations["P(t)"]


    draw_f_chart(data, linspace, calculations["f(t)"])
    draw_q_chart(sorted_data, linspace, calculations["Q*(t)"], calculations["Q(t)"])
    draw_p_chart(sorted_data, linspace, calculations["P*(t)"], calculations["P(t)"])
    draw_lambda_chart(linspace, calculations["Lambda(t)"])

if __name__ == "__main__":
    app()
